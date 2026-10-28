#!/bin/bash
# macOS: double-click to start the Clausa web interface and open it in the browser.
# Stop it with Ctrl+C in this window, or close the window.
cd "$(dirname "$0")" || exit 1
( sleep 4; open "http://127.0.0.1:7860" ) &
exec ./clausa webui
