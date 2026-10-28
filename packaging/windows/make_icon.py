#!/usr/bin/env python3
"""Make icon.ico, the Windows icon, from the app window's icon.

The icon is runner/app_ui/icon.png (1024 px; packaging/macos/make_icon.py makes
the Mac's .icns from it). This scales it with sips (macOS) to the sizes Windows
asks for and writes runner/app_ui/icon.ico: the window's and the taskbar's icon
(runner/app.py) and, compiled in by packaging/build_windows.py, Clausa.exe's.
Sizes up to 64 px are 32-bit bitmaps, 256 px is PNG (both readable by Windows
and by .NET's System.Drawing.Icon, which pywebview uses). icon.ico is committed,
so the bundle build needs neither; run this again after changing icon.png.

    python3 packaging/windows/make_icon.py
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ICON = ROOT / "runner" / "app_ui" / "icon.png"
OUT = ROOT / "runner" / "app_ui" / "icon.ico"
BITMAP_SIZES = (16, 20, 24, 32, 40, 48, 64)
PNG_SIZE = 256


def rgba(png: bytes) -> tuple[int, int, bytes]:
    """Width, height and RGBA rows of an 8-bit RGBA or RGB, non-interlaced PNG (as sips writes)."""
    pos, idat, width = 8, b"", 0
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos:pos + 8])
        data = png[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", data)
            if depth != 8 or color not in (2, 6) or interlace:
                raise SystemExit(f"unexpected PNG format (depth {depth}, color type {color}, interlace {interlace})")
            channels = 4 if color == 6 else 3
        elif kind == b"IDAT":
            idat += data
        pos += 12 + length
    raw, stride, out, prev = zlib.decompress(idat), width * channels, bytearray(), bytearray(width * channels)
    for y in range(height):
        kind, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b, c = prev[i], prev[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + a) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + b) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        prev = line
        if channels == 3:
            line = bytearray(b"".join(bytes(line[i:i + 3]) + b"\xff" for i in range(0, stride, 3)))
        out += line
    return width, height, bytes(out)


def bitmap_entry(png: bytes) -> bytes:
    """A 32-bit BGRA icon bitmap: BITMAPINFOHEADER (height doubled), rows bottom-up, AND mask."""
    w, h, pixels = rgba(png)
    rows = []
    for y in range(h - 1, -1, -1):
        row = pixels[y * w * 4:(y + 1) * w * 4]
        rows.append(b"".join(bytes((row[i + 2], row[i + 1], row[i], row[i + 3])) for i in range(0, len(row), 4)))
    mask_stride = ((w + 31) // 32) * 4
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, w * h * 4 + mask_stride * h, 0, 0, 0, 0)
    return header + b"".join(rows) + b"\x00" * (mask_stride * h)


def main() -> int:
    entries = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in (*BITMAP_SIZES, PNG_SIZE):
            out = Path(tmp) / f"{size}.png"
            subprocess.run(["sips", "-s", "format", "png", "-z", str(size), str(size), str(ICON), "--out", str(out)],
                           check=True, capture_output=True)
            data = out.read_bytes()
            entries.append((size, data if size == PNG_SIZE else bitmap_entry(data)))
    offset = 6 + 16 * len(entries)
    head = struct.pack("<HHH", 0, 1, len(entries))
    body = b""
    for size, data in entries:
        head += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset + len(body))
        body += data
    OUT.write_bytes(head + body)
    print(f"Wrote {OUT} ({len(entries)} sizes, {OUT.stat().st_size:,} bytes) from {ICON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
