"""Tkinter GUI: [ Stream 1 live view ] [ Stream 2 live view ] [ Controls ]."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from dataclasses import replace
from datetime import date, datetime
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from . import app_dir
from .config import Config, ConfigError, load_config, save_config
from .error_log import get_log_path, log_error
from .event_tracker import EventTracker
from .excel_logger import ExcelLogError, ExcelLogger, create_workbook
from .stream_monitor import StreamMonitor, fmt_duration

VIEW_W, VIEW_H = 560, 315        # initial size of each live view (16:9)
FRAME_MS = 66                    # live view refresh (~15 fps)
TICK_MS = 1000                   # counters / Excel update
SAVE_EVERY_S = 30                # periodic Excel save while logging


class StreamView(ttk.LabelFrame):
    """One live-view section: a video canvas plus today's counters."""

    def __init__(self, master, title: str):
        super().__init__(master, text=title, padding=6)
        self.canvas = tk.Canvas(self, width=VIEW_W, height=VIEW_H, bg="black",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        # keep the "STOPPED" text centred when the window is resized
        self.canvas.bind("<Configure>", lambda e: self.canvas.coords(
            "stopped", e.width // 2, e.height // 2))
        self.stats = ttk.Label(self, text="Today: 0 events  |  0:00:00 blocked",
                               font=("Segoe UI", 11, "bold"))
        self.stats.pack(anchor="w", pady=(6, 0))
        self._photo = None
        self._image_id = None
        self.show_stopped()

    def view_size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)

    def show_stopped(self) -> None:
        self.canvas.delete("all")
        self._photo = self._image_id = None
        w, h = self.view_size() if self.canvas.winfo_ismapped() else (VIEW_W, VIEW_H)
        self.canvas.create_text(w // 2, h // 2, text="STOPPED", fill="white",
                                font=("Segoe UI", 28, "bold"), tags="stopped")

    def show_frame(self, rgb) -> None:
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        if self._image_id is None:
            self.canvas.delete("all")
            self._image_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfigure(self._image_id, image=self._photo)

    def set_stats(self, events: int, seconds: float) -> None:
        self.stats.config(text=f"Today: {events} event{'s' if events != 1 else ''}  |  "
                               f"{fmt_duration(seconds)} blocked")


AUTOSTART_RETRY_S = 60           # retry interval if an automatic start fails


class App(tk.Tk):
    def __init__(self, autostart: bool = False):
        super().__init__()
        self.title("IsHrbtBlocked - HRBT Tunnel Monitor")
        self.minsize(1100, 450)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running = False
        self.logger: ExcelLogger | None = None
        self.trackers: dict[int, EventTracker] = {}
        self.monitors: dict[int, StreamMonitor] = {}
        self._last_save = 0.0
        self._frame_job = None
        self._tick_job = None

        self.cfg, created, config_problem = self._load_config()
        self._build_ui()
        self._fill_fields()

        if config_problem:
            self.after(300, lambda: messagebox.showerror(
                "Settings error",
                f"{config_problem}\n\nDefault settings have been loaded. Correct the "
                "fields and click 'Update Settings' to rewrite config.json.", parent=self))
        elif autostart and not created:
            # launched by the boot script: start logging without anyone clicking
            self.after(1500, self._auto_start)
        elif created:
            self.after(300, lambda: messagebox.showwarning(
                "Update stream URLs",
                "config.json was not found, so a new one was created with placeholder "
                "settings.\n\nPlease enter the Stream 1 and Stream 2 URLs and click "
                "'Update Settings' before starting logging.", parent=self))

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_config() -> tuple[Config, bool, str | None]:
        try:
            cfg, created = load_config()
            return cfg, created, None
        except ConfigError as exc:
            log_error(f"config.json problem: {exc}")
            return Config(), False, str(exc)
        except OSError as exc:
            log_error("Could not create config.json.", exc)
            return Config(), False, f"Could not create config.json: {exc}"

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Start.TButton", font=("Segoe UI", 10, "bold"))

        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=0)
        root.rowconfigure(0, weight=1)

        # Sections 1 and 2: live views
        self.views = {
            1: StreamView(root, self.cfg.stream_1_name),
            2: StreamView(root, self.cfg.stream_2_name),
        }
        self.views[1].grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.views[2].grid(row=0, column=1, sticky="nsew", padx=(0, 6))

        # Section 3: controls
        ctl = ttk.LabelFrame(root, text="Controls", padding=10)
        ctl.grid(row=0, column=2, sticky="nsew")
        ctl.columnconfigure(0, weight=1)

        btns = ttk.Frame(ctl)
        btns.grid(row=0, column=0, sticky="ew")
        btns.columnconfigure((0, 1), weight=1)
        self.start_btn = ttk.Button(btns, text="Start Logging", style="Start.TButton",
                                    command=self.start_logging)
        self.stop_btn = ttk.Button(btns, text="Stop Logging", command=self.stop_logging,
                                   state="disabled")
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_btn.grid(row=0, column=1, sticky="ew")

        self.state_lbl = ttk.Label(ctl, text="Logging: STOPPED", foreground="#b00020",
                                   font=("Segoe UI", 10, "bold"))
        self.state_lbl.grid(row=1, column=0, sticky="w", pady=(8, 4))
        ttk.Separator(ctl).grid(row=2, column=0, sticky="ew", pady=6)

        self.var_s1 = tk.StringVar()
        self.var_s2 = tk.StringVar()
        self.var_thr = tk.StringVar()
        self.var_start = tk.StringVar()
        self.var_end = tk.StringVar()

        row = 3
        for label, var, width in (
            ("Stream 1 URL", self.var_s1, 38),
            ("Stream 2 URL", self.var_s2, 38),
            ("Event threshold (seconds with no vehicle)", self.var_thr, 10),
        ):
            ttk.Label(ctl, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(ctl, textvariable=var, width=width).grid(row=row + 1, column=0,
                                                              sticky="ew" if width > 10 else "w",
                                                              pady=(0, 8))
            row += 2

        ttk.Label(ctl, text="Logging window (24-hour, HHMM)").grid(row=row, column=0, sticky="w")
        win = ttk.Frame(ctl)
        win.grid(row=row + 1, column=0, sticky="w", pady=(0, 8))
        ttk.Label(win, text="Start").pack(side="left")
        ttk.Entry(win, textvariable=self.var_start, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(win, text="End").pack(side="left")
        ttk.Entry(win, textvariable=self.var_end, width=6).pack(side="left", padx=(4, 0))
        row += 2

        ttk.Button(ctl, text="Update Settings", command=self.update_settings).grid(
            row=row, column=0, sticky="ew", pady=(4, 8))
        row += 1
        ttk.Separator(ctl).grid(row=row, column=0, sticky="ew", pady=6)
        row += 1
        self.info_lbl = ttk.Label(ctl, text="", wraplength=280, justify="left",
                                  foreground="#555")
        self.info_lbl.grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Button(ctl, text="Open Error Log", command=self.open_error_log).grid(
            row=row, column=0, sticky="w", pady=(8, 0))
        self._update_info()

    def _fill_fields(self) -> None:
        self.var_s1.set(self.cfg.stream_1_url)
        self.var_s2.set(self.cfg.stream_2_url)
        self.var_thr.set(str(self.cfg.event_threshold_seconds))
        self.var_start.set(self.cfg.window_start)
        self.var_end.set(self.cfg.window_end)

    def _update_info(self) -> None:
        self.info_lbl.config(text=(
            f"An event is logged when no vehicle passes for "
            f"{self.cfg.event_threshold_seconds}s between {self.cfg.window_start} "
            f"and {self.cfg.window_end}.\n\nLog file: {self.cfg.excel_file}"))

    # ----------------------------------------------------------- start / stop
    def _auto_start(self) -> None:
        """Start logging unattended (after a reboot). Never waits on a dialog:
        a missing Excel file is created automatically, and on failure it
        retries every minute until it succeeds or someone clicks Start."""
        if self.running:
            return
        if self.cfg.streams_are_placeholders():
            log_error("Autostart: stream URLs are not set - logging not started.")
            messagebox.showwarning(
                "Update stream URLs",
                "The app was started automatically, but the stream URLs are not set.\n\n"
                "Enter them, click 'Update Settings', then 'Start Logging'.", parent=self)
            return
        if self.start_logging(unattended=True):
            log_error("Autostart: logging started.", level="INFO")
        else:
            self.info_lbl.config(text=f"Automatic start failed (see errors.log). "
                                      f"Retrying in {AUTOSTART_RETRY_S} s...")
            self.after(AUTOSTART_RETRY_S * 1000, self._auto_start)

    def start_logging(self, unattended: bool = False) -> bool:
        """Open the Excel log and start both streams. Returns True on success.

        unattended=True (autostart): no dialogs - a missing Excel file is
        created automatically and errors only go to errors.log.
        """
        if self.running:
            return True
        if self.cfg.streams_are_placeholders():
            messagebox.showwarning(
                "Update stream URLs",
                "Please enter valid Stream 1 and Stream 2 URLs and click "
                "'Update Settings' before starting.", parent=self)
            return False

        path = app_dir() / self.cfg.excel_file
        if not path.exists() and unattended:
            try:
                create_workbook(path)
                log_error(f"Autostart: {path.name} was missing and has been created.",
                          level="INFO")
            except Exception as exc:
                log_error(f"Could not create {path.name}.", exc)
                return False
        elif not path.exists():
            create = messagebox.askyesno(
                "Excel log not found",
                f"The Excel log file '{path.name}' was not found.\n\n"
                "Do you want to create it?", parent=self)
            if not create:
                messagebox.showinfo("Logging aborted", "Logging aborted.", parent=self)
                return False
            try:
                create_workbook(path)
            except Exception as exc:
                log_error(f"Could not create {path.name}.", exc)
                messagebox.showerror("Excel error", f"Could not create {path.name}:\n{exc}",
                                     parent=self)
                return False

        logger = ExcelLogger(path)
        try:
            logger.open()
        except ExcelLogError as exc:
            log_error(str(exc), exc.__cause__)
            if not unattended:
                messagebox.showerror("Excel error", str(exc), parent=self)
            return False
        self.logger = logger
        self._begin_monitoring()
        self._update_info()
        return True

    def _begin_monitoring(self) -> None:
        c = self.cfg
        start_min, end_min = c.window_minutes()
        model_path = app_dir() / c.model
        self.trackers = {i: EventTracker(c.event_threshold_seconds, start_min, end_min)
                         for i in (1, 2)}
        self.monitors = {
            i: StreamMonitor(i, name, url, self.trackers[i], model_path, c.confidence,
                             c.motion_pixel_fraction, c.ignore_stationary_vehicles,
                             c.stale_stream_seconds, c.min_vehicle_size,
                             c.detection_image_size)
            for i, name, url in ((1, c.stream_1_name, c.stream_1_url),
                                 (2, c.stream_2_name, c.stream_2_url))
        }
        for m in self.monitors.values():
            m.start()
        self.running = True
        self._last_save = datetime.now().timestamp()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.state_lbl.config(text="Logging: RUNNING", foreground="#1b7f2a")
        self._frame_loop()
        self._tick()

    def stop_logging(self) -> None:
        """Close the streams, write everything to Excel and release the file."""
        if not self.running:
            return
        for job in (self._frame_job, self._tick_job):
            if job:
                self.after_cancel(job)
        self._frame_job = self._tick_job = None

        for m in self.monitors.values():
            m.stop()
        for t in self.trackers.values():
            t.end_event()
        self._flush_trackers(flush=True)

        if self.logger:
            while not self.logger.close():
                retry = messagebox.askretrycancel(
                    "Excel save failed",
                    f"Could not save '{self.logger.path.name}'. If it is open in Excel, "
                    "close it and click Retry.\n\nCancel discards unsaved counts from "
                    "this session.", parent=self)
                if not retry:
                    log_error("User discarded unsaved Excel data after a failed save.")
                    self.logger.discard()
                    break
        self.logger = None
        self.monitors, self.trackers = {}, {}
        self.running = False

        for v in self.views.values():
            v.show_stopped()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.state_lbl.config(text="Logging: STOPPED", foreground="#b00020")

    # ---------------------------------------------------------------- loops
    def _frame_loop(self) -> None:
        for i, m in self.monitors.items():
            view = self.views[i]
            w, h = view.view_size()
            try:
                rgb = m.render(w, h, self.cfg.show_detection_boxes)
                if rgb is not None:
                    view.show_frame(rgb)
            except Exception as exc:
                log_error(f"{m.name}: display error.", exc)
        self._frame_job = self.after(FRAME_MS, self._frame_loop)

    def _flush_trackers(self, flush: bool = False) -> bool:
        new_event = False
        if not self.logger:
            return False
        for i, t in self.trackers.items():
            new_event |= self.logger.add(i, t.drain(flush=flush))
        return new_event

    def _tick(self) -> None:
        try:
            today = date.today()
            new_event = self._flush_trackers()
            now = datetime.now().timestamp()
            if new_event or now - self._last_save >= SAVE_EVERY_S:
                self.logger.ensure_row(today)
                self.logger.save()
                self._last_save = now
            totals = self.logger.totals(today)
            for i, (ev_idx, sec_idx) in ((1, (0, 1)), (2, (2, 3))):
                p_ev, p_sec = self.trackers[i].pending(today)
                self.views[i].set_stats(totals[ev_idx] + p_ev, totals[sec_idx] + p_sec)
        except Exception as exc:
            log_error("Error while updating counters.", exc)
        self._tick_job = self.after(TICK_MS, self._tick)

    # -------------------------------------------------------------- settings
    def update_settings(self) -> None:
        new_cfg = replace(
            self.cfg,
            stream_1_url=self.var_s1.get(),
            stream_2_url=self.var_s2.get(),
            event_threshold_seconds=self.var_thr.get(),
            window_start=self.var_start.get(),
            window_end=self.var_end.get(),
        )
        try:
            new_cfg.validate()
        except ConfigError as exc:
            messagebox.showerror("Invalid setting", str(exc), parent=self)
            return

        was_running = self.running
        if was_running:                       # pause monitoring
            self.stop_logging()
        try:
            save_config(new_cfg)
        except OSError as exc:
            log_error("Could not save config.json.", exc)
            messagebox.showerror("Settings error", f"Could not save config.json:\n{exc}",
                                 parent=self)
            return
        self.cfg = new_cfg
        self._fill_fields()
        self._update_info()
        if was_running:                       # reload live monitoring
            self.start_logging()
        else:
            messagebox.showinfo("Settings updated", "Settings saved to config.json.",
                                parent=self)

    # ------------------------------------------------------------------ misc
    def open_error_log(self) -> None:
        path = get_log_path()
        if not path.exists():
            messagebox.showinfo("Error log", "No errors have been logged yet.", parent=self)
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(path)])
        except Exception as exc:
            messagebox.showinfo("Error log", f"Error log location:\n{path}\n\n({exc})", parent=self)

    def report_callback_exception(self, exc_type, exc, tb):  # Tk callback errors
        log_error("Unhandled GUI error.", exc)
        messagebox.showerror("Unexpected error", f"{exc}\n\nDetails were written to errors.log.",
                             parent=self)

    def on_close(self) -> None:
        if self.running:
            self.stop_logging()
        self.destroy()
