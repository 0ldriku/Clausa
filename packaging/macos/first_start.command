#!/bin/bash
# "First start.command": double-clicked once after unzipping (packaging/build_macos.py).
#
# 1. Checks the Mac (Apple Silicon, the minimum macOS) and the folder (not in
#    a folder that syncs to the cloud).
# 2. Clears the quarantine flag that macOS puts on everything in a downloaded
#    folder, so that Clausa.app runs from here and can use envs/.
# 3. Unpacks the first environment, qwen3-asr (the app's Python): the zip
#    holds each environment as one tar (envs/<name>.tar), because Archive
#    Utility handles a zip of their 115,000 files very slowly. A tar unpacks
#    into envs/.unpacking-<name>, is moved into place and then deleted, so the
#    disk needs room for one environment twice at most (checked first).
# 4. Opens the First start window (runner/first_start_window.py), which shows
#    the progress in the app's design, and runs the rest with qwen3-asr's
#    Python (runner/first_start_macos.py): it unpacks the other environments,
#    downloads the speech models into cache/ (the pinned versions, each file
#    checked against its sha256; a zip built with --include-models has them
#    already), loads the main libraries once so that macOS checks them now,
#    and writes envs/.first-start-done (Clausa.app refuses to start without it).
# 5. Opens Clausa.app when the window's "Open Clausa" is clicked.
#
# Both report to logs/first-start-events.jsonl, which the window reads
# (docs/development.md, "The app zips"); this
# script writes only fixed strings and numbers there. The window tells this
# script what the user chose by its exit code. Without the window (it could
# not open), everything is reported here in Terminal, as before.
#
# Running it again is harmless: what is done is skipped. After an interruption
# it starts the unfinished environment again from its tar, which is deleted
# only once that environment is in place, and continues a model download
# where it stopped.
#
# For tests: CLAUSA_FIRST_START_NO_OPEN=1 skips opening Clausa.app,
# CLAUSA_FIRST_START_NO_WINDOW=1 the window, and CLAUSA_FIRST_START_TEST=<folder>
# makes the window keep snapshots and take clicks (runner/first_start_window.py). HF_ENDPOINT (a
# Hugging Face mirror), CLAUSA_MFA_BASE_URL (another address for the MFA
# files; runner/models.py) and the usual proxy variables are passed on to the
# download.

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd -P)"
E="$ROOT/envs"
LOG="$ROOT/logs/first-start.log"
MARKER="$E/.first-start-done"
EV="$ROOT/logs/first-start-events.jsonl"
WINRC="$ROOT/logs/.first-start-window.$$.rc"  # this run's window (a rerun keeps the pid)
LOCK="$ROOT/logs/first-start.lock"
VERSION="{{VERSION}}"

fail() {  # stop here; with a message ($1) when it was not shown yet
  if [ -n "$1" ]; then
    echo "Stopped: $1" >> "$LOG" 2>/dev/null
    echo
    echo "Stopped: $1"
  fi
  echo
  echo "Details are in logs/first-start.log in the Clausa folder. If you cannot"
  echo "fix this, please send that file with a description of the problem."
  echo
  read -r -p "Press Return to close this window. "
  exit 1
}
ev() {  # one event: fixed strings and numbers only (runner/first_start_events.py)
  printf '{"t":%s,%s}\n' "$(date +%s)" "$1" >> "$EV"
}
gb() { echo "$(( ($1 + 1048575) / 1048576 ))"; }  # KB to GB, rounded up
gb_down() { echo "$(( $1 / 1048576 ))"; }

[ -t 1 ] && clear
echo "Clausa: first start"
echo "Do not move or rename the Clausa folder until Clausa says it is ready."
echo

mkdir -p "$ROOT/logs" 2>/dev/null && touch "$LOG" 2>/dev/null \
  || fail "Clausa cannot write in its folder ($ROOT). Move the Clausa folder into your home folder and double-click First start again."
printf '\n%s First start in %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$ROOT" >> "$LOG"
# One First start at a time (a double-click while one runs); shlock clears a
# lock whose process is gone.
if ! /usr/bin/shlock -p $$ -f "$LOCK"; then
  echo "First start is already running (see the Clausa window or the other"
  echo "Terminal window)."
  echo
  read -r -p "Press Return to close this window. "
  exit 1
fi
trap 'rm -f "$LOCK"' EXIT

# This Mac must be able to run the environments (the builder fills in the
# highest minimum macOS version of their libraries).
MIN_MACOS="{{MIN_MACOS}}"
if [ "$(uname -m)" != arm64 ]; then
  fail "Clausa needs a Mac with Apple Silicon (M1 or later); this Mac has an Intel processor."
fi
IFS=. read -r have_major have_minor _ <<< "$(sw_vers -productVersion)"
IFS=. read -r need_major need_minor _ <<< "$MIN_MACOS"
if [ "$have_major" -lt "$need_major" ] || { [ "$have_major" -eq "$need_major" ] && [ "${have_minor:-0}" -lt "${need_minor:-0}" ]; }; then
  fail "Clausa needs macOS $MIN_MACOS or later; this Mac has macOS $(sw_vers -productVersion). Please update macOS first."
fi
# Syncing 13 GB of program files is slow and can remove files that Clausa needs.
case "$ROOT/" in
  */Library/Mobile\ Documents/*|*/Library/CloudStorage/*|"$HOME"/Dropbox/*)
    fail "the Clausa folder is in a folder that syncs to the cloud (iCloud Drive, Dropbox, OneDrive or Google Drive). Move the Clausa folder into your home folder (in Finder: Go > Home), then double-click First start again." ;;
esac

echo "Allowing macOS to open the app's files..."
/usr/bin/xattr -drs com.apple.quarantine "$ROOT" 2>> "$LOG"  # -s: symlinks too
if /usr/bin/xattr -p com.apple.quarantine "$ROOT/Clausa.app" >/dev/null 2>&1; then
  fail "the quarantine flag could not be removed from Clausa.app."
fi

# The run's start: the window shows what follows it. A run after an
# unfinished one (its events file is still there) says it continues.
if [ -s "$EV" ] || [ -f "$MARKER" ]; then resumed=true; else resumed=false; fi
ev '"event":"start","run":"'"$(date +%s)-$$"'","platform":"macos","version":"'"$VERSION"'","pid":'"$$"',"resumed":'"$resumed"',"steps":[{"id":"unpack","label":"Unpacking the app"},{"id":"download","label":"Downloading the speech models"},{"id":"prepare","label":"Preparing (only once)"}]'

# The first environment: the window runs with its Python. A tar left over
# from an interrupted run is unpacked again from the start.
PY="$E/qwen3-asr/bin/python"
tar="$E/qwen3-asr.tar"
if [ -f "$tar" ]; then
  size=$(stat -f %z "$tar")
  tar_kb=$(( size / 1024 ))
  need_kb=$(( tar_kb + 2 * 1048576 ))  # the unpacked environment, plus 2 GB to spare
  free_kb=$(df -Pk "$E" | awk 'NR == 2 {print $4}')
  if [ "$free_kb" -lt "$need_kb" ]; then
    echo "  not enough space for qwen3-asr: $free_kb KB free, $need_kb KB needed" >> "$LOG"
    fail "not enough free disk space. Unpacking needs about $(gb "$need_kb") GB free on this disk; there are $(gb_down "$free_kb") GB. Free some space (you can delete the downloaded zip), then double-click First start again."
  fi
  ev '"event":"step","step":"unpack","status":"running"'
  ev '"event":"item","step":"unpack","item":"qwen3-asr","label":"The app and speech recognition","total":'"$size"',"status":"running","done":0'
  echo "Unpacking the app (the first part, $(gb "$tar_kb") GB)..."
  start=$(date +%s)
  rm -rf "$E/.unpacking-qwen3-asr" && mkdir "$E/.unpacking-qwen3-asr" \
    || fail "Clausa cannot write in envs/."
  if ! /usr/bin/tar -xf "$tar" -C "$E/.unpacking-qwen3-asr" 2>> "$LOG"; then
    rm -rf "$E/.unpacking-qwen3-asr"
    fail "the app could not be unpacked (is the disk full?). Free some space, then double-click First start again."
  fi
  rm -rf "$E/qwen3-asr" && mv "$E/.unpacking-qwen3-asr/qwen3-asr" "$E/qwen3-asr" && rmdir "$E/.unpacking-qwen3-asr" \
    || fail "qwen3-asr could not be moved into place in envs/."
  /usr/bin/xattr -drs com.apple.quarantine "$E/qwen3-asr" 2>> "$LOG"
  rm -f "$tar"
  ev '"event":"item","step":"unpack","item":"qwen3-asr","label":"The app and speech recognition","total":'"$size"',"status":"done","done":'"$size"
  echo "  qwen3-asr unpacked in $(( $(date +%s) - start )) s" >> "$LOG"
elif [ ! -x "$PY" ]; then
  fail "the environment qwen3-asr is missing (neither envs/qwen3-asr.tar nor envs/qwen3-asr). Please unzip the download again."
fi

# The rest runs with qwen3-asr's Python, in a clean environment.
pass=()
for v in HF_ENDPOINT CLAUSA_MFA_BASE_URL CLAUSA_FIRST_START_TEST HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy NO_PROXY no_proxy; do
  [ -n "${!v}" ] && pass+=("$v=${!v}")
done
py() {  # call it in a subshell, ( py ... ): it replaces that process with Python, so $! is Python's pid
  cd "$ROOT/app" && exec env -i HOME="$HOME" PATH="$E/qwen3-asr/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
    CONDA_PREFIX="$E/qwen3-asr" CLAUSA_ENVS_DIR="$E" CLAUSA_CACHE="$ROOT/cache" CLAUSA_LOGS="$ROOT/logs" \
    "${pass[@]}" "$PY" "$@"
}

window_closed() {  # the window has closed (USR1)
  [ -f "$WINRC" ] || return 0
  case "$(cat "$WINRC")" in
    1)  # before First start was done: stop the rest
      [ -n "$CHILD" ] || return 0
      kill -TERM "$CHILD" 2>/dev/null
      wait "$CHILD" 2>/dev/null
      echo "  the window was closed before First start was done: stopped" >> "$LOG"
      echo
      echo "First start was stopped: its window was closed. Double-click First"
      echo "start again to continue; it continues where it stopped."
      exit 1 ;;
    0|3|75) ;;  # answers after done or an error: read where this script waits for them
    *)  # 10: it could not open; or it failed: everything is reported here
      [ "$WINDOW" = open ] && echo "The progress window could not open; First start goes on here."
      WINDOW=none ;;
  esac
}
trap window_closed USR1

wait_window() {  # sets WCODE to the window's exit code, once it has closed
  while [ ! -f "$WINRC" ]; do
    wait "$WIN_JOB" 2>/dev/null || sleep 0.2
    # its job gone without a code (the disk full, say): as if it could not open
    kill -0 "$WIN_JOB" 2>/dev/null || [ -f "$WINRC" ] || break
  done
  WCODE="$(cat "$WINRC" 2>/dev/null)"
  WCODE="${WCODE:-10}"
}

# The window, in the background. It ignores Terminal's hangup, so that when
# this Terminal window is closed it can say that First start stopped. When
# it closes, it leaves its exit code in $WINRC and signals this script.
WINDOW=none
CHILD=""
rm -f "$ROOT"/logs/.first-start-window.*.rc "$ROOT/logs/first-start-window.pid"
if [ -z "$CLAUSA_FIRST_START_NO_WINDOW" ]; then
  ( trap '' HUP
    ( py -m runner.first_start_window --root "$ROOT" ) >> "$LOG" 2>&1
    echo $? > "$WINRC.part" && mv "$WINRC.part" "$WINRC"
    kill -USR1 $$ 2>/dev/null ) &
  WIN_JOB=$!
  WINDOW=open
  echo "Progress is shown in the Clausa window."
fi

( py -m runner.first_start_macos --root "$ROOT" ) 2>> "$LOG" &
CHILD=$!  # Python itself: SIGTERM reaches its handler, which stops its own children
while :; do
  wait "$CHILD"
  rc=$?
  # A status above 128 with the process still there: the wait was cut short
  # by the window's signal.
  [ $rc -gt 128 ] && kill -0 "$CHILD" 2>/dev/null && continue
  break
done
CHILD=""
if [ $rc -eq 1 ] || [ $rc -gt 5 ]; then
  # it ended without saying why (Python itself failed, or it was stopped)
  echo "  runner.first_start_macos ended with $rc" >> "$LOG"
  ev '"event":"error","step":null,"kind":"system","again":true,"message":"First start stopped unexpectedly. Try again; if it happens again, please send the log."'
  [ "$WINDOW" = none ] && echo "Stopped: First start stopped unexpectedly (exit $rc)."
fi

if [ $rc -ne 0 ]; then
  [ "$WINDOW" = open ] || fail ""
  echo "First start stopped: the Clausa window says why and what to do."
  wait_window
  case "$WCODE" in
    75)
      echo "Trying again..." && echo
      wait "$WIN_JOB" 2>/dev/null  # its signal must not reach the new run, which has no trap yet
      rm -f "$LOCK"
      exec /bin/bash "$ROOT/First start.command" ;;
    0|1|3) ;;
    *)  # the window failed: say here what it would have said
      fail "$(grep -E '"event": ?"error"' "$EV" 2>/dev/null | tail -1 | sed -n -E 's/.*"message": ?"([^"]*)".*/\1/p')" ;;
  esac
  echo
  echo "Double-click First start again to continue; it continues where it stopped."
  exit 1
fi

WCODE=0
if [ "$WINDOW" = open ]; then
  echo "Ready: see the Clausa window."
  wait_window
fi
rm -f "$EV" "$WINRC"  # the progress of a finished First start is not needed again
echo
if [ "$WCODE" = 3 ] || [ "$WCODE" = 1 ]; then
  echo "Clausa is ready. From now on, double-click Clausa.app to start it."
  echo "You can close this window."
  exit 0
fi
if [ -n "$CLAUSA_FIRST_START_NO_OPEN" ]; then
  echo "Ready (CLAUSA_FIRST_START_NO_OPEN is set: not opening Clausa)."
  exit 0
fi
echo "Opening Clausa. From now on, double-click Clausa.app to start it."
/usr/bin/open "$ROOT/Clausa.app" || fail "Clausa.app could not be opened."

# The window stays open: closing it from here would mean scripting Terminal
# (osascript), which can make macOS ask whether Terminal may control Terminal.
echo "You can close this window."
exit 0
