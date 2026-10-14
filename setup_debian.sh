#!/usr/bin/env bash
# =============================================================================
#  IsHrbtBlocked - Debian desktop setup
#
#  Makes the app open and start logging by itself every time the VM boots:
#    1. installs the system packages the app needs (Tkinter, OpenCV libraries)
#    2. creates / reuses the Python virtual environment and installs packages
#    3. checks your settings from the first launch (config.json, Excel, model)
#    4. offers to set the time zone to America/New_York
#    5. writes run_hrbt.sh, a launcher that waits for the network, starts the
#       app with --autostart and restarts it if it ever crashes
#    6. adds the launcher to your desktop's autostart (+ an app-menu entry)
#    7. turns on automatic login, so no one has to log in after a reboot
#    8. turns off suspend / hibernate, so the VM never goes to sleep
#
#  Run it as the desktop user who will be logged in (NOT with sudo):
#      chmod +x setup_debian.sh
#      ./setup_debian.sh            # interactive (recommended)
#      ./setup_debian.sh --yes      # accept every default without asking
#      ./setup_debian.sh --uninstall
#
#  Safe to run again; it only changes what isn't already set up.
# =============================================================================
set -euo pipefail

APP_NAME="IsHrbtBlocked"
APP_ID="ishrbtblocked"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$APP_DIR/run_hrbt.sh"
AUTOSTART_FILE="$HOME/.config/autostart/$APP_ID.desktop"
MENU_FILE="$HOME/.local/share/applications/$APP_ID.desktop"
TARGET_TZ="America/New_York"
LIGHTDM_DROPIN="/etc/lightdm/lightdm.conf.d/50-$APP_ID-autologin.conf"
SDDM_DROPIN="/etc/sddm.conf.d/50-$APP_ID-autologin.conf"
SLEEP_TARGETS="sleep.target suspend.target hibernate.target hybrid-sleep.target"

ASSUME_YES=0
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

# ----------------------------------------------------------------- helpers
bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
fail() { printf '\n\033[31mERROR:\033[0m %s\n' "$*"; exit 1; }

# ask "question" Y|N  -> returns 0 for yes
ask() {
    local q="$1" def="${2:-Y}" reply prompt="[Y/n]"
    [ "$def" = "N" ] && prompt="[y/N]"
    if [ "$ASSUME_YES" = 1 ]; then [ "$def" = "Y" ]; return; fi
    read -r -p "   $q $prompt " reply || reply=""
    reply="${reply:-$def}"
    [[ "$reply" =~ ^[Yy] ]]
}

display_manager() {
    local dm=""
    if [ -f /etc/X11/default-display-manager ]; then
        dm="$(basename "$(cat /etc/X11/default-display-manager)")"
    elif [ -L /etc/systemd/system/display-manager.service ]; then
        dm="$(basename "$(readlink -f /etc/systemd/system/display-manager.service)" .service)"
    fi
    echo "$dm"
}

# ----------------------------------------------------------------- checks
[ "$(id -u)" -ne 0 ] || fail "Run this as your normal desktop user, not with sudo / as root.
       (The script asks for your password when it needs admin rights.)"
command -v sudo >/dev/null || fail "sudo is not installed. As root run: apt install sudo && adduser $USER sudo, then log out and in."
[ -f "$APP_DIR/main.py" ] || fail "main.py not found next to this script ($APP_DIR)."
[ -f /etc/debian_version ] || warn "This doesn't look like Debian - continuing anyway."
USER_NAME="$(id -un)"

# =========================================================== uninstall mode
if [ "$UNINSTALL" = 1 ]; then
    bold "Removing $APP_NAME autostart"
    rm -f "$AUTOSTART_FILE" "$MENU_FILE" "$LAUNCHER"
    ok "Removed autostart entry, menu entry and run_hrbt.sh"
    if ask "Turn automatic login back off?" Y; then
        sudo rm -f "$LIGHTDM_DROPIN" "$SDDM_DROPIN"
        if [ -f /etc/gdm3/daemon.conf ]; then
            sudo sed -i -E 's/^[#[:space:]]*AutomaticLoginEnable[[:space:]]*=.*/AutomaticLoginEnable=false/' /etc/gdm3/daemon.conf
        fi
        ok "Automatic login turned off"
    fi
    if ask "Allow the VM to suspend / hibernate again?" Y; then
        # shellcheck disable=SC2086
        sudo systemctl unmask $SLEEP_TARGETS >/dev/null 2>&1 || true
        ok "Sleep re-enabled"
    fi
    echo
    echo "Done. Your settings, Excel log and the virtual environment were left in place."
    exit 0
fi

echo "Setting up $APP_NAME in: $APP_DIR"
echo "Desktop user:           $USER_NAME"
sudo -v || fail "Admin rights (sudo) are needed."

# ================================================= 1. system packages
bold "1/8  System packages"
PKGS=(python3 python3-venv python3-pip python3-tk libgl1 libglib2.0-0)
MISSING=()
for p in "${PKGS[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || MISSING+=("$p")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "   Installing: ${MISSING[*]}"
    sudo apt-get update -q
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${MISSING[@]}"
fi
ok "python3, venv, Tkinter and OpenCV libraries installed"

# ================================================= 2. virtual environment
bold "2/8  Python virtual environment"
VENV="$APP_DIR/.venv"
if [ -d "$VENV" ] && [ ! -x "$VENV/bin/python" ]; then
    # a .venv copied over from Windows (has Scripts\ instead of bin/) - leave it alone
    warn ".venv looks like a Windows environment; using .venv-linux instead"
    VENV="$APP_DIR/.venv-linux"
fi
if [ ! -x "$VENV/bin/python" ]; then
    echo "   Creating $VENV"
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -c "import tkinter" 2>/dev/null || fail "Tkinter isn't available in $VENV.
       Delete that folder and run this script again."

if "$PY" -c "import torch, ultralytics, cv2, openpyxl, PIL, numpy" 2>/dev/null; then
    ok "Python packages already installed ($VENV)"
else
    echo "   Installing Python packages (CPU-only PyTorch - this takes a few minutes)..."
    "$PY" -m pip install -q --upgrade pip
    if ! "$PY" -c "import torch" 2>/dev/null; then
        "$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    "$PY" -m pip install -q -r "$APP_DIR/requirements.txt"
    "$PY" -c "import torch, ultralytics, cv2, openpyxl, PIL, numpy" \
        || fail "Package install didn't complete - see the messages above."
    ok "Python packages installed ($VENV)"
fi

# ================================================= 3. first-launch files
bold "3/8  Settings from the first launch"
SETTINGS_OK=1
if [ ! -f "$APP_DIR/config.json" ]; then
    warn "config.json not found - launch the app once (\"$PY\" main.py), set the"
    warn "stream URLs, click Update Settings, then run this script again."
    SETTINGS_OK=0
else
    (cd "$APP_DIR" && "$PY" - <<'PYEOF'
import sys
from hrbt.config import load_config
try:
    cfg, _ = load_config()
except Exception as exc:
    print(f"   config.json problem: {exc}"); sys.exit(2)
if cfg.streams_are_placeholders():
    print("   config.json still has placeholder stream URLs"); sys.exit(3)
print(f"   streams: {cfg.stream_1_url}\n            {cfg.stream_2_url}")
print(f"   window {cfg.window_start}-{cfg.window_end}, threshold {cfg.event_threshold_seconds}s")
PYEOF
    ) && ok "config.json is set up" || { warn "Fix config.json (open the app, Update Settings) - autostart can't log until then."; SETTINGS_OK=0; }
    EXCEL_NAME="$(cd "$APP_DIR" && "$PY" -c 'from hrbt.config import load_config; print(load_config()[0].excel_file)' 2>/dev/null || echo hrbt_event_log.xlsx)"
    MODEL_NAME="$(cd "$APP_DIR" && "$PY" -c 'from hrbt.config import load_config; print(load_config()[0].model)' 2>/dev/null || echo yolov8s.pt)"
    if [ -f "$APP_DIR/$EXCEL_NAME" ]; then ok "Excel log found ($EXCEL_NAME)"
    else warn "$EXCEL_NAME not found - it will be created automatically at the first autostart"; fi
    if [ -f "$APP_DIR/$MODEL_NAME" ]; then ok "YOLO model found ($MODEL_NAME)"
    else warn "$MODEL_NAME not found - it will download at the first start (needs internet)"; fi
fi

# ================================================= 4. time zone
bold "4/8  Time zone"
CURRENT_TZ="$(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || echo unknown)"
if [ "$CURRENT_TZ" = "$TARGET_TZ" ]; then
    ok "Time zone is $TARGET_TZ"
else
    warn "Time zone is $CURRENT_TZ. The logging window and Excel dates use local time."
    if ask "Set the time zone to $TARGET_TZ?" Y; then
        sudo timedatectl set-timezone "$TARGET_TZ"
        ok "Time zone set to $TARGET_TZ"
    fi
fi

# ================================================= 5. launcher
bold "5/8  Launcher (run_hrbt.sh)"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Generated by setup_debian.sh - starts $APP_NAME with logging switched on.
# Waits for the network, then keeps the app running: if it crashes it is
# restarted. Closing the window yourself stops it until the next login.
APP_DIR="$APP_DIR"
PY="$PY"
EOF
cat >> "$LAUNCHER" <<'EOF'
LOG="$APP_DIR/launcher.log"
cd "$APP_DIR" || exit 1

if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 1048576 ]; then mv -f "$LOG" "$LOG.1"; fi
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
trap 'log "launcher stopped (session ending)"; [ -n "${child:-}" ] && kill "$child" 2>/dev/null; exit 0' TERM INT HUP

log "launcher started (DISPLAY=${DISPLAY:-} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-})"
sleep "${HRBT_START_DELAY:-10}"            # let the desktop finish loading

# wait up to 2 minutes for the network / DNS (the app also reconnects by itself)
for _ in $(seq 1 60); do
    getent hosts media-sfs1.vdotcameras.com >/dev/null 2>&1 && break
    sleep 2
done

quick_crashes=0
while true; do
    started=$(date +%s)
    "$PY" "$APP_DIR/main.py" --autostart >> "$LOG" 2>&1 &
    child=$!
    wait "$child"
    code=$?
    if [ "$code" -eq 0 ]; then
        log "app closed normally - not restarting"
        exit 0
    fi
    if [ $(( $(date +%s) - started )) -lt 120 ]; then
        quick_crashes=$((quick_crashes + 1))
    else
        quick_crashes=0
    fi
    delay=15
    [ "$quick_crashes" -ge 5 ] && delay=300   # crashing over and over - back off
    log "app exited with code $code - restarting in ${delay}s (see errors.log)"
    sleep "$delay"
done
EOF
chmod +x "$LAUNCHER"
ok "Wrote $LAUNCHER"

# ================================================= 6. autostart + menu entry
bold "6/8  Desktop autostart"
mkdir -p "$(dirname "$AUTOSTART_FILE")" "$(dirname "$MENU_FILE")"
cat > "$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=HRBT tunnel-exit stoppage monitor (starts logging automatically)
Exec=/bin/bash "$LAUNCHER"
Path=$APP_DIR
Terminal=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
Hidden=false
EOF
cat > "$MENU_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=HRBT tunnel-exit stoppage monitor
Exec=/bin/bash "$LAUNCHER"
Path=$APP_DIR
Terminal=false
Categories=Utility;
EOF
ok "Autostart entry: $AUTOSTART_FILE"
ok "App-menu entry:  $MENU_FILE (starts it with logging on)"

# ================================================= 7. automatic login
bold "7/8  Automatic login"
DM="$(display_manager)"
echo "   Display manager: ${DM:-unknown}"
echo "   Without automatic login the app only starts after someone logs in."
if ask "Log $USER_NAME in automatically at boot?" Y; then
    case "$DM" in
        gdm3|gdm)
            CONF=/etc/gdm3/daemon.conf
            [ -f "$CONF" ] || { sudo mkdir -p /etc/gdm3; printf '[daemon]\n' | sudo tee "$CONF" >/dev/null; }
            [ -f "$CONF.$APP_ID.bak" ] || sudo cp "$CONF" "$CONF.$APP_ID.bak"
            sudo python3 - "$CONF" "$USER_NAME" <<'PYEOF'
import re, sys
path, user = sys.argv[1], sys.argv[2]
text = open(path).read()
if not re.search(r"^\[daemon\]", text, re.M):
    text += "\n[daemon]\n"
for key, val in (("AutomaticLogin", user), ("AutomaticLoginEnable", "true")):
    pat = re.compile(rf"^[#\s]*{key}\s*=.*$", re.M)
    if pat.search(text):
        text = pat.sub(f"{key}={val}", text, count=1)
    else:
        text = re.sub(r"^\[daemon\][^\n]*\n", lambda m: m.group(0) + f"{key}={val}\n",
                      text, count=1, flags=re.M)
open(path, "w").write(text)
PYEOF
            ok "GDM: automatic login enabled for $USER_NAME (backup: $CONF.$APP_ID.bak)"
            ;;
        lightdm)
            sudo mkdir -p "$(dirname "$LIGHTDM_DROPIN")"
            printf '[Seat:*]\nautologin-user=%s\nautologin-user-timeout=0\n' "$USER_NAME" \
                | sudo tee "$LIGHTDM_DROPIN" >/dev/null
            if getent group autologin >/dev/null; then sudo usermod -aG autologin "$USER_NAME"; fi
            ok "LightDM: automatic login enabled for $USER_NAME"
            ;;
        sddm)
            SESSION=""
            for s in plasma plasmax11 plasmawayland lxqt; do
                if [ -f "/usr/share/xsessions/$s.desktop" ] || [ -f "/usr/share/wayland-sessions/$s.desktop" ]; then
                    SESSION="$s"; break
                fi
            done
            if [ -z "$SESSION" ]; then
                SESSION="$(basename "$(ls /usr/share/xsessions/*.desktop /usr/share/wayland-sessions/*.desktop 2>/dev/null | head -n1)" .desktop)"
            fi
            sudo mkdir -p "$(dirname "$SDDM_DROPIN")"
            printf '[Autologin]\nUser=%s\nSession=%s\n' "$USER_NAME" "$SESSION" \
                | sudo tee "$SDDM_DROPIN" >/dev/null
            ok "SDDM: automatic login enabled for $USER_NAME (session: $SESSION)"
            ;;
        *)
            warn "Couldn't recognise the display manager - turn on automatic login in"
            warn "your desktop's Settings > Users (or your login manager's settings)."
            ;;
    esac
else
    warn "Skipped - the app will start when $USER_NAME logs in."
fi

# ================================================= 8. no sleep
bold "8/8  Sleep / suspend"
echo "   A suspended VM stops monitoring, and some desktops suspend after ~15 min idle."
if ask "Turn off suspend and hibernate?" Y; then
    # shellcheck disable=SC2086
    sudo systemctl mask $SLEEP_TARGETS >/dev/null
    ok "Suspend and hibernate disabled (screen blanking/locking is fine - the app keeps running)"
else
    warn "Skipped - make sure the VM never suspends, or logging will stop."
fi

# ================================================= summary
bold "Done"
echo "   On every boot: auto-login -> $APP_NAME opens -> logging starts by itself."
echo "   If it crashes it restarts. Closing the window stops it until the next login"
echo "   (start it again from the app menu: $APP_NAME)."
echo "   Logs: $APP_DIR/errors.log  and  $APP_DIR/launcher.log"
[ "$SETTINGS_OK" = 1 ] || warn "Remember to finish the settings (see step 3) before rebooting."
echo
if ask "Reboot now to test it?" N; then
    sudo systemctl reboot
else
    echo "   Reboot whenever you're ready:  sudo reboot"
fi
