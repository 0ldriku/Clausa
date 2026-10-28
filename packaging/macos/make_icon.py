#!/usr/bin/env python3
"""Make Clausa.icns, the Mac app's icon, from the app window's icon.

The icon is runner/app_ui/icon.png (1024 px, rendered from icon.html with
headless Chrome; the window uses it as its Dock icon). This scales it to the
ten sizes of an iconset with sips and builds the .icns with iconutil (macOS).
Clausa.icns is committed, so the bundle build needs neither; run this again
after changing icon.png.

    python3 packaging/macos/make_icon.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICON = HERE.parents[1] / "runner" / "app_ui" / "icon.png"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "Clausa.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(["sips", "-z", str(px), str(px), str(ICON), "--out", str(iconset / name)],
                               check=True, capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "Clausa.icns")], check=True)
    print(f"Wrote {HERE / 'Clausa.icns'} from {ICON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
