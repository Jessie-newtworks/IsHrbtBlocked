"""Tests for footage recording (hrbt/recorder.py) and its settings.
Run:  python -m pytest

The ffmpeg test runs only if ffmpeg is installed; the rest need no ffmpeg."""

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from hrbt import error_log, recorder
from hrbt.config import Config, ConfigError
from hrbt.event_tracker import EventTracker, TrackerSnapshot
from hrbt.recorder import (GB, FootageRecorder, Segment, SegmentRecorder, build_ffmpeg_cmd,
                           clip_filename, enforce_cap, join_segments, list_segments,
                           parse_segment_time, prune_old_segments, recording_active,
                           select_segments)
from hrbt.stream_monitor import StreamMonitor

URL1 = "https://example.com/1/playlist.m3u8"
URL2 = "https://example.com/2/playlist.m3u8"


@pytest.fixture(autouse=True)
def _tmp_error_log(tmp_path, monkeypatch):
    error_log.set_log_path(tmp_path / "errors.log")
    # plenty of disk unless a test says otherwise (the real disk may be nearly full)
    monkeypatch.setattr(recorder, "free_bytes", lambda path: 100 * GB)


def errors(tmp_path) -> str:
    p = tmp_path / "errors.log"
    return p.read_text() if p.exists() else ""


def ts(h, m=0, s=0, day=10):
    return datetime(2026, 11, day, h, m, s).timestamp()


def make_segment(folder: Path, start: float, size: int = 100, length: float = 60,
                 key: str = "stream1", fill: bytes = b"x") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{key}_{datetime.fromtimestamp(start):%Y-%m-%d_%H-%M-%S}.ts"
    p = folder / name
    p.write_bytes(fill * size)
    os.utime(p, (start + length, start + length))
    return p


def cfg(**kw):
    c = Config(stream_1_url=URL1, stream_2_url=URL2, **kw)
    return c.validate()


# ---------------------------------------------------------------- settings
def test_recording_defaults():
    c = Config().validate()
    assert c.record_footage is True and c.recording_folder == "recordings"
    assert c.recording_segment_seconds == 300 and c.recording_keep_days == 5
    assert c.event_clips_max_gb == 3 and c.min_free_disk_gb == 5
    assert c.event_clip_pre_seconds == 60 and c.event_clip_post_seconds == 120
    assert c.near_miss_clips is False and c.near_miss_min_seconds == 15


@pytest.mark.parametrize("field,value", [
    ("recording_keep_days", 0), ("recording_keep_days", -1), ("recording_keep_days", "x"),
    ("recording_keep_days", 400), ("event_clips_max_gb", 0), ("min_free_disk_gb", -1),
    ("event_clip_pre_seconds", -5), ("event_clip_post_seconds", 601),
    ("recording_segment_seconds", 5), ("recording_segment_seconds", 30.5),
    ("recording_segment_seconds", 601), ("record_footage", "maybe"),
    ("recording_folder", "  "), ("near_miss_clips_max_gb", 0), ("event_clips_max_gb", True),
])
def test_bad_recording_settings(field, value):
    with pytest.raises(ConfigError):
        Config(**{field: value}).validate()


def test_recording_settings_are_normalised():
    c = Config(record_footage="false", recording_keep_days="7", event_clips_max_gb="2.5",
               min_free_disk_gb=0, recording_segment_seconds=60.0).validate()
    assert c.record_footage is False and c.recording_keep_days == 7
    assert c.event_clips_max_gb == 2.5 and c.min_free_disk_gb == 0
    assert c.recording_segment_seconds == 60 and isinstance(c.recording_segment_seconds, int)


def test_near_miss_must_be_below_threshold():
    with pytest.raises(ConfigError):
        Config(near_miss_clips=True, near_miss_min_seconds=30,
               event_threshold_seconds=30).validate()
    Config(near_miss_clips=False, near_miss_min_seconds=30).validate()   # unused: fine


def test_old_config_file_gets_recording_keys(tmp_path):
    import json
    from hrbt.config import load_config
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"stream_1_url": URL1, "stream_2_url": URL2}))
    c, created = load_config(p)
    assert not created and c.record_footage
    assert json.loads(p.read_text())["recording_keep_days"] == 5


# ------------------------------------------------------------------ helpers
def test_recording_window_includes_pre_and_post_roll():
    d = lambda h, m=0: datetime(2026, 11, 10, h, m)  # noqa: E731
    assert recording_active(d(12), 360, 1200)
    assert not recording_active(d(5, 58), 360, 1200, 60, 120)
    assert recording_active(d(5, 59), 360, 1200, 60, 120)       # pre-roll before 0600
    assert recording_active(d(20, 1), 360, 1200, 60, 120)       # post-roll after 2000
    assert not recording_active(d(20, 3), 360, 1200, 60, 120)
    assert recording_active(d(2), 1320, 360) and not recording_active(d(12), 1320, 360)
    assert recording_active(d(3), 0, 0)                         # all day


def test_segment_names_round_trip(tmp_path):
    p = make_segment(tmp_path, ts(6, 1, 2))
    assert p.name == "stream1_2026-11-10_06-01-02.ts"
    assert parse_segment_time(p) == ts(6, 1, 2)
    assert parse_segment_time(tmp_path / "notes.txt") is None
    assert parse_segment_time(tmp_path / "stream1_2026-13-10_06-01-02.ts") is None
    (tmp_path / "junk.ts").write_bytes(b"1")
    segs = list_segments(tmp_path)
    assert [s.path for s in segs] == [p] and segs[0].end == ts(6, 2, 2)


def test_ffmpeg_command_saves_without_reencoding(tmp_path):
    cmd = build_ffmpeg_cmd("ffmpeg", URL1, tmp_path / "50%", "stream1", 60)
    assert cmd[cmd.index("-c") + 1] == "copy" and cmd[cmd.index("-i") + 1] == URL1
    assert cmd[cmd.index("-segment_time") + 1] == "60"
    assert cmd[-1].endswith("stream1_%Y-%m-%d_%H-%M-%S.ts") and "50%%" in cmd[-1]


def test_select_segments():
    segs = [Segment(Path(f"s{i}"), ts(7, i), ts(7, i + 1), 10) for i in range(5)]
    segs.append(Segment(Path("empty"), ts(7, 5), ts(7, 6), 0))
    picked = select_segments(segs, ts(7, 1, 30), ts(7, 3, 10))
    assert [s.path.name for s in picked] == ["s1", "s2", "s3"]
    assert select_segments(segs, ts(8), ts(9)) == []


def test_clip_filename_matches_events_log():
    assert clip_filename("Stream 2 HRBT EB Exit", "event", ts(7, 41, 3), 192.4) == \
        "2026-11-10_07-41-03_Stream_2_HRBT_EB_Exit_event_0h03m12s.ts"
    assert clip_filename("Stream 1 HRBT WB Exit", "gap", ts(7, 0), 22.6) == \
        "2026-11-10_07-00-00_Stream_1_HRBT_WB_Exit_gap_23s.ts"


def test_join_segments(tmp_path):
    a = make_segment(tmp_path / "f", ts(7, 0), 3, fill=b"a")
    b = make_segment(tmp_path / "f", ts(7, 1), 2, fill=b"b")
    dest = tmp_path / "clips" / "c.ts"
    segs = list_segments(tmp_path / "f")
    assert join_segments(segs, dest) == 5 and dest.read_bytes() == b"aaabb"
    assert not list((tmp_path / "clips").glob("*.part"))
    assert a.exists() and b.exists()


def test_enforce_cap_deletes_oldest_first(tmp_path):
    for i in range(5):
        (tmp_path / f"2026-11-0{i + 1}_07-00-00_S_event_0h01m00s.ts").write_bytes(b"x" * 100)
    deleted = enforce_cap(tmp_path, 250)
    assert [p.name[:10] for p in deleted] == ["2026-11-01", "2026-11-02", "2026-11-03"]
    assert sorted(p.name[:10] for p in tmp_path.iterdir()) == ["2026-11-04", "2026-11-05"]
    (tmp_path / "2026-11-06_07-00-00_S_event_9h00m00s.ts").write_bytes(b"x" * 1000)
    enforce_cap(tmp_path, 250)                                   # newest is always kept
    assert [p.name[:10] for p in tmp_path.iterdir()] == ["2026-11-06"]


def test_prune_old_segments(tmp_path):
    full = tmp_path / "full"
    old = [make_segment(full / "stream1", ts(7, i, day=4)) for i in range(3)]
    new = make_segment(full / "stream1", ts(7, 0, day=9))
    other = make_segment(full / "stream2", ts(7, 0, day=4), key="stream2")
    assert prune_old_segments(full, ts(0, day=5)) == 4
    assert new.exists() and not any(p.exists() for p in old + [other])


# ------------------------------------------------------------ FootageRecorder
@pytest.fixture
def fake_ffmpeg(monkeypatch):
    monkeypatch.setattr(recorder.shutil, "which", lambda name: "/usr/bin/ffmpeg")


def test_missing_ffmpeg_is_logged_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder.shutil, "which", lambda name: None)
    r = FootageRecorder(cfg(), tmp_path)
    r.start()
    r.add_clip(1, "event", ts(7), ts(7, 5), 300)
    r.stop()
    assert "ffmpeg was not found" in errors(tmp_path) and "sudo apt install -y ffmpeg" in \
        errors(tmp_path)
    assert "OFF" in r.status and not (tmp_path / "recordings").exists()


def test_event_clip_is_cut_from_segments(tmp_path, fake_ffmpeg):
    r = FootageRecorder(cfg(), tmp_path)
    folder = tmp_path / "recordings" / "full" / "stream2"
    for m in range(0, 15):
        make_segment(folder, ts(7, m), 10, key="stream2", fill=bytes([65 + m]))
    # event 07:05:30 - 07:07:10, pre 60 s / post 120 s -> 07:04:30 to 07:09:10
    r.add_clip(2, "event", ts(7, 5, 30), ts(7, 7, 10), 100)
    r._process_clips(ts(7, 9, 15))                       # too early: post-roll not done
    assert len(r._jobs) == 1
    r._process_clips(ts(7, 30))
    clip = tmp_path / "recordings" / "events" / \
        "2026-11-10_07-05-30_Stream_2_HRBT_EB_Exit_event_0h01m40s.ts"
    assert clip.exists() and not r._jobs
    assert clip.read_bytes() == b"".join(bytes([65 + m]) * 10 for m in range(4, 10))


def test_near_miss_clips_only_when_enabled(tmp_path, fake_ffmpeg):
    r = FootageRecorder(cfg(), tmp_path)
    r.add_clip(1, "gap", ts(7), ts(7, 0, 20), 20)
    assert not r._jobs
    r = FootageRecorder(cfg(near_miss_clips=True), tmp_path)
    make_segment(tmp_path / "recordings" / "full" / "stream1", ts(7))
    r.add_clip(1, "gap", ts(7, 0, 10), ts(7, 0, 30), 20)
    r.stop()                                     # stop writes waiting clips at once
    assert [p.name for p in (tmp_path / "recordings" / "near_misses").iterdir()] == \
        ["2026-11-10_07-00-10_Stream_1_HRBT_WB_Exit_gap_20s.ts"]


def test_missing_footage_is_a_warning(tmp_path, fake_ffmpeg):
    r = FootageRecorder(cfg(), tmp_path)
    r.add_clip(1, "event", ts(7), ts(7, 5), 300)
    r.stop()
    assert "no footage recorded" in errors(tmp_path)


def test_clip_cap_is_enforced(tmp_path, fake_ffmpeg):
    r = FootageRecorder(cfg(event_clips_max_gb=250 / GB), tmp_path)
    folder = tmp_path / "recordings" / "full" / "stream1"
    for h in (7, 8, 9):
        make_segment(folder, ts(h), 100)
        r.add_clip(1, "event", ts(h, 0, 10), ts(h, 0, 50), 40)
    r.stop()
    names = sorted(p.name[:13] for p in (tmp_path / "recordings" / "events").iterdir())
    assert names == ["2026-11-10_08", "2026-11-10_09"]


class FakeDisk:
    """free_bytes() that goes up as recordings are deleted."""

    def __init__(self, root: Path, free: float):
        self.root, self.base = root, free
        self.start = self.used()

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob("*.ts"))

    def __call__(self, path):
        return int(self.base + self.start - self.used())


def test_disk_guard_prunes_oldest_then_pauses(tmp_path, fake_ffmpeg, monkeypatch):
    c = cfg(min_free_disk_gb=1000 / GB)
    r = FootageRecorder(c, tmp_path)
    folder = tmp_path / "recordings" / "full" / "stream1"
    segs = [make_segment(folder, ts(7, m), 300) for m in range(6)]
    disk = FakeDisk(tmp_path / "recordings", 500)
    monkeypatch.setattr(recorder, "free_bytes", disk)
    r.housekeeping(ts(12))
    # needed 500 more bytes: the two oldest segments go, the rest stay
    assert [p.exists() for p in segs] == [False, False, True, True, True, True]
    assert not r.disk_low

    disk.base = -10_000                                   # something else filled the disk
    r.housekeeping(ts(12, 1))
    assert r.disk_low and "PAUSED" in r.status and "Footage recording PAUSED" in errors(tmp_path)
    assert not r._want_recording(1, ts(12, 1))
    disk.base = 10_000 + 2 * GB                            # room again (over the margin)
    r.housekeeping(ts(12, 2))
    assert not r.disk_low and "resumed" in errors(tmp_path)


def test_disk_guard_keeps_recent_and_needed_footage(tmp_path, fake_ffmpeg, monkeypatch):
    r = FootageRecorder(cfg(min_free_disk_gb=10 ** 6 / GB), tmp_path)
    folder = tmp_path / "recordings" / "full" / "stream1"
    needed = make_segment(folder, ts(7), 300)
    recent = make_segment(folder, ts(11, 55), 300)
    other = make_segment(folder, ts(8), 300)
    monkeypatch.setattr(recorder, "free_bytes", FakeDisk(tmp_path / "recordings", 0))
    r.add_clip(1, "event", ts(7, 0, 10), ts(7, 0, 40), 30)
    r.housekeeping(ts(12))
    assert needed.exists() and recent.exists() and not other.exists()
    assert r.disk_low


def test_retention_deletes_old_days(tmp_path, fake_ffmpeg, monkeypatch):
    r = FootageRecorder(cfg(recording_keep_days=5), tmp_path)
    folder = tmp_path / "recordings" / "full" / "stream1"
    old = make_segment(folder, ts(19, day=4))
    keep = make_segment(folder, ts(13, day=5))
    monkeypatch.setattr(recorder, "free_bytes", lambda p: 100 * GB)
    r.housekeeping(ts(12, day=10))
    assert not old.exists() and keep.exists()


def test_recording_runs_on_for_post_roll(tmp_path, fake_ffmpeg):
    r = FootageRecorder(cfg(), tmp_path)                 # window 0600-2000, post 120 s
    assert r._want_recording(1, ts(12)) and not r._want_recording(1, ts(20, 5))
    r.add_clip(1, "event", ts(20, 4), ts(20, 4, 30), 30)
    assert r._want_recording(1, ts(20, 5)) and not r._want_recording(2, ts(20, 5))


# -------------------------------------------------------- ffmpeg supervision
@pytest.mark.skipif(sys.platform.startswith("win"), reason="uses a shell script as ffmpeg")
def test_ffmpeg_crash_is_logged_and_restarted(tmp_path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\necho 'Server returned 404 Not Found' >&2\nexit 1\n")
    fake.chmod(0o755)
    rec = SegmentRecorder(1, "Stream 1", URL1, tmp_path / "full" / "stream1", str(fake), 60)
    rec.supervise(1000.0, True)
    rec.proc.wait(5)
    rec.supervise(1001.0, True)                          # notices the exit
    assert rec.proc is None
    log = errors(tmp_path)
    assert "exit code 1" in log and "404 Not Found" in log
    rec.supervise(1003.0, True)                          # waits RESTART_DELAY_S first
    assert rec.proc is None
    rec.supervise(1007.0, True)
    assert rec.proc is not None                          # restarted
    rec.stop()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="uses a shell script as ffmpeg")
def test_stalled_ffmpeg_is_restarted(tmp_path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nexec sleep 600\n")
    fake.chmod(0o755)
    rec = SegmentRecorder(1, "Stream 1", URL1, tmp_path / "full" / "stream1", str(fake), 60)
    rec.supervise(1000.0, True)
    proc = rec.proc
    for t in range(1015, 1080, 15):
        rec.supervise(float(t), True)
    assert rec.proc is None and proc.poll() is not None
    assert "stalled" in errors(tmp_path)
    rec.supervise(1100.0, False)


def _has_ffmpeg_x264() -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True)
    return "libx264" in out.stdout


@pytest.mark.skipif(not _has_ffmpeg_x264(), reason="needs ffmpeg with libx264")
def test_real_ffmpeg_writes_segments(tmp_path):
    src = tmp_path / "src.ts"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x240:rate=15", "-t", "4", "-c:v", "libx264",
                    "-g", "15", "-f", "mpegts", str(src)], check=True, timeout=60)
    out = tmp_path / "full" / "stream1"
    out.mkdir(parents=True)
    cmd = build_ffmpeg_cmd("ffmpeg", str(src), out, "stream1", 10)
    subprocess.run(cmd, check=True, timeout=60)
    segs = list_segments(out)
    assert segs and all(s.size > 0 for s in segs)
    assert abs(segs[0].start - datetime.now().timestamp()) < 120
    clip = tmp_path / "clip.ts"
    join_segments(segs, clip)
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(clip), "-f", "null", "-"],
                           capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0 and "h264" in probe.stderr


# ------------------------------------------------- StreamMonitor -> clips
def snap(clear=0.0, in_event=False, feed_ok=True, in_win=True, event_s=0.0):
    return TrackerSnapshot(feed_ok, in_win, in_event, clear, event_s)


def monitor_with_clips(near_miss=15):
    tr = EventTracker(30, 360, 1200)
    m = StreamMonitor(2, "Stream 2 HRBT EB Exit", URL2, tr, Path("x.pt"), 0.25, 0.0005,
                      True, 20)
    calls = []
    m.on_clip = lambda *a: calls.append(a)
    m.near_miss_seconds = near_miss
    return m, calls


def test_event_end_requests_clip():
    m, calls = monitor_with_clips()
    m._log_event_change(snap(clear=31, in_event=True, event_s=31), 1000.0)
    m._log_event_change(snap(clear=90, in_event=True, event_s=90), 1059.0)
    m._log_event_change(snap(), 1060.0)
    assert calls == [(2, "event", 1000.0, 1060.0, 90)]


def test_near_miss_gap_requests_clip():
    m, calls = monitor_with_clips()
    m._log_event_change(snap(clear=10), 1010.0)
    m._log_event_change(snap(), 1011.0)                   # 10 s gap: too short
    m._log_event_change(snap(clear=22), 1033.0)
    m._log_event_change(snap(), 1034.0)                   # 22 s gap: near miss
    m._log_event_change(snap(clear=22, in_win=False), 1056.0)
    m._log_event_change(snap(), 1057.0)                   # outside the window
    m._log_event_change(snap(clear=22), 1079.0)
    m._log_event_change(snap(feed_ok=False), 1080.0)      # feed lost, not a car
    assert calls == [(2, "gap", 1034.0 - 22, 1034.0, 22)]


def test_near_miss_off_by_default():
    m, calls = monitor_with_clips(near_miss=0)
    m._log_event_change(snap(clear=22), 1033.0)
    m._log_event_change(snap(), 1034.0)
    assert calls == []


def test_stop_during_event_requests_clip():
    m, calls = monitor_with_clips()
    m._log_event_change(snap(clear=40, in_event=True, event_s=40), 1000.0)
    m._log_event_change(snap(clear=70, in_event=True, event_s=70), 1030.0)
    m.stop()
    assert calls == [(2, "event", 1000.0, 1030.0, 70)]


def test_clip_callback_error_never_breaks_detection(tmp_path):
    m, _ = monitor_with_clips()
    m.on_clip = lambda *a: 1 / 0
    m._log_event_change(snap(clear=31, in_event=True, event_s=31), 1000.0)
    m._log_event_change(snap(), 1001.0)
    assert "could not queue a footage clip" in errors(tmp_path)
