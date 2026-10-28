#!/bin/bash
# Clausa.app/Contents/MacOS/Clausa: starts the app window with the environments
# in the app's folder (packaging/build_macos.py puts it there).
#
# The folder holds Clausa.app, envs/, cache/, app/ (the code) and logs/. Until
# "First start" has cleared the quarantine flag, macOS runs an app from a
# downloaded folder from a read-only copy elsewhere (App Translocation), where
# envs/ is out of reach; and until it has finished, envs/ is not unpacked.
# In both cases this asks for First start and exits.

APP_DIR="$(cd "$(dirname "$0")/../.." && pwd -P)"
ROOT="$(dirname "$APP_DIR")"
LOG="$ROOT/logs/launcher.log"

log() {
  mkdir -p "$ROOT/logs" 2>/dev/null
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG" 2>/dev/null
}

dialog() {
  /usr/bin/osascript - "$1" >/dev/null 2>&1 <<'EOF'
on run argv
  display dialog (item 1 of argv) with title "Clausa" buttons {"OK"} default button "OK" with icon caution
end run
EOF
}

FIRST_START="Clausa is not set up yet. Please double-click “First start” in the Clausa folder and wait until its window says that Clausa is ready (a few minutes, only once). After that, Clausa.app opens with a double-click."

case "$APP_DIR" in
  */AppTranslocation/*)
    dialog "$FIRST_START"
    exit 0 ;;
esac
if [ ! -f "$ROOT/envs/.first-start-done" ]; then  # written when First start has finished
  log "First start has not finished: asked for it"
  dialog "$FIRST_START"
  exit 0
fi
for f in "$APP_DIR" "$ROOT/envs/qwen3-asr/bin/python"; do
  if /usr/bin/xattr -p com.apple.quarantine "$f" >/dev/null 2>&1; then
    log "quarantine flag still set on $f: asked for First start"
    dialog "$FIRST_START"
    exit 0
  fi
done

ENV="$ROOT/envs/qwen3-asr"
if [ ! -x "$ENV/bin/python" ] || [ ! -f "$ROOT/app/runner/app.py" ]; then
  log "incomplete folder: $ENV/bin/python or $ROOT/app/runner/app.py is missing"
  dialog "Clausa cannot start: files are missing from its folder ($ROOT). Please unzip the download again."
  exit 1
fi

# What `./clausa app` sets up (runner/cli.py, runner/envs.py env_vars), done
# here so that the window belongs to this process, which macOS launched as
# Clausa.app (its name and icon in the Dock).
export CLAUSA_ENVS_DIR="$ROOT/envs"
export CLAUSA_CACHE="$ROOT/cache"
export CLAUSA_LOGS="$ROOT/logs"
export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:/usr/bin:/bin:/usr/sbin:/sbin"
unset PYTHONPATH PYTHONHOME
cd "$ROOT/app" || { log "cannot enter $ROOT/app"; exit 1; }
# A failure to start shows a dialog instead of a Dock icon that bounces and goes.
if ! out="$("$ENV/bin/python" -c 'import webview, runner.app' 2>&1)"; then
  log "cannot start the app: $out"
  dialog "Clausa could not start. Details are in logs/launcher.log in the Clausa folder; please send that file with a description of the problem."
  exit 1
fi
log "starting Clausa from $ROOT"
# The app keeps its own log (logs/app-<date>.log and each run's log.txt); the
# console, which repeats it, is kept for the last start only, for crashes.
exec "$ENV/bin/python" -m runner.app > "$ROOT/logs/console.log" 2>&1
