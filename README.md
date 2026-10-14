# IsHrbtBlocked

Watches the VDOT 511 live cameras at the Hampton Roads Bridge-Tunnel exits and uses
YOLOv8s to spot traffic stoppages. A stoppage ("event") is logged when no vehicle
passes the camera for a set number of seconds during the logging window.

## Setup (Windows)

```powershell
cd C:\Users\jessi\Documents\Projects\coding_projects\IsHrbtBlocked
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org) and
YOLO will use it automatically. On a CPU, the motion check keeps the load down. If the
live view shows "(detector behind)", the CPU can't keep up during heavy traffic. That
usually doesn't affect stoppage detection, because frames back up only when there's a lot of motion (i.e. traffic flowing).
`yolov8s.pt` downloads automatically the first time you start logging (about 22 MB).

> If `pip install` can't find a `torch` build for your Python version, make the venv with
> Python 3.12 instead.

## First run

1. On launch the app looks for `config.json`. If it's missing, the app creates it with
   placeholder streams, `0600`–`2000` and a 30-second threshold, then warns you to update
   the streams.
2. Paste the stream URLs into the Controls panel and click **Update Settings**:
   - Westbound: `https://media-sfs8.vdotcameras.com/rtplive/HamptonRoads1014/playlist.m3u8`
   - Eastbound: `https://media-sfs1.vdotcameras.com/rtplive/HRBT1018/playlist.m3u8`
3. Click **Start Logging**. If the Excel log doesn't exist, the app asks whether to create
   it. Clicking No shows "Logging aborted".

## How events work

- **Every frame** from each camera is checked for cars, motorcycles, buses and trucks.
  These cameras run at about 15 fps or less, so a fast car may be in view for only 2–3
  frames. A quick motion check runs first, and YOLO is skipped on frames where nothing
  changed. While the exit is empty, YOLO hardly runs at all.
- Timing uses the video's own frame timestamps, so the buffering in HLS streams doesn't
  distort the gaps between cars.
- When no vehicle has passed for **threshold** seconds inside the window, that day's event
  counter goes up by 1. Blocked time is measured from when the road went empty, so it
  includes the threshold seconds. The default threshold is 30 s. Normal gaps between
  cars at these exits reach 5–10 s (longer in light traffic), and a real blockage lasts
  minutes.
- The timer pauses when a vehicle passes again, and stays paused until the next event.
- Both cameras watch tunnel **exits**. When traffic is stopped further back in the tunnel,
  no cars come out, so a long gap with no passing car means a stoppage.
- By default only **moving** vehicles count as passing (`ignore_stationary_vehicles`), so a
  parked or broken-down vehicle sitting in view can't hide a stoppage. On the live view, green boxes are moving
  vehicles and orange boxes are stopped ones. A vehicle counts as stopped only if its box
  stayed in place at every check over the last 2 seconds (0.5, 1, 1.5 and 2 s back). That
  way distant cars, and cars driving straight toward the camera, still count as moving.
  The test allows for YOLO's boxes wobbling by a few pixels.
- **Grey boxes** are vehicles too far away to judge reliably (smaller than
  `min_vehicle_size`, default 6 % of the picture height). They're ignored. Every car
  leaving the tunnel passes close to the camera, so it's counted there.
- If a stream drops, or no new frames arrive for `stale_stream_seconds`, that time is
  **not** counted as blocked. The app
  reconnects automatically and writes the failure to `errors.log`.
- Windows that cross midnight work (e.g. `2200`–`0600`). If start and end are the same,
  the window is all day.

## Files (next to `main.py`)

| File | Purpose |
|---|---|
| `config.json` | Settings. Edit them in the GUI or by hand |
| `hrbt_event_log.xlsx` | A: Date · B: Stream 1 events · C: Stream 1 seconds blocked · D: Stream 2 events · E: Stream 2 seconds blocked |
| `errors.log` | Plain-text error log (also notes each camera's frame rate after connecting) |
| `events.log` | One line per event start and end, with times and duration. Use it to check the daily totals against what really happened |

The Excel file is saved within a second of each new event, every 30 s while logging, and
when logging stops. If you have it open in Excel, the save is retried automatically.
When you click Stop, the app asks you to close Excel so it can write the final save.

### Advanced settings (config.json only)

`stream_1_name`, `stream_2_name`, `excel_file`, `model`, `confidence`,
`motion_pixel_fraction` (0 = run YOLO on every frame), `ignore_stationary_vehicles`, `stale_stream_seconds`,
`show_detection_boxes`, `min_vehicle_size`, `detection_image_size`. The top of `hrbt/config.py` describes each one.

## Running on a Debian VM (starts by itself after every reboot)

1. Copy the project folder to the VM (e.g. `~/IsHrbtBlocked`). You can bring
   `config.json`, the Excel log and `yolov8s.pt` along. A Windows `.venv` is ignored.
2. First launch: set up the streams and settings as usual. If the VM doesn't have the
   Python packages yet, run `./setup_debian.sh` first, then
   `.venv/bin/python main.py`, then run the setup script once more.
3. Run the setup script as your normal desktop user (not with sudo):

   ```bash
   cd ~/IsHrbtBlocked
   chmod +x setup_debian.sh
   ./setup_debian.sh
   ```

   It installs the system packages (Tkinter, OpenCV libraries) and CPU-only PyTorch, and
   checks your settings. It sets the time zone to America/New_York and creates `run_hrbt.sh`.
   It also adds the app to the desktop's autostart, turns on automatic login (GDM,
   LightDM or SDDM), and turns off suspend. Each change asks first. `--yes` accepts all
   of them.
4. Reboot. The desktop logs in, the app opens, and logging starts with no clicks.

How it behaves after a reboot:
- `main.py --autostart` starts logging by itself. If the Excel log is missing, it's
  created without asking. If a start fails (e.g. the Excel file can't be opened), it
  retries every minute.
- If the app crashes, `run_hrbt.sh` restarts it after 15 s. Closing the window yourself
  stops it until the next login. The app-menu entry "IsHrbtBlocked" starts it again.
- Only one copy can run at a time, so two copies never write the same Excel file.
- Logs: `errors.log` (the app) and `launcher.log` (starts and restarts).
- To undo everything: `./setup_debian.sh --uninstall`

## Tests

```powershell
python -m pytest
```
