"""Keep the computer from going to sleep while the app analyses recordings.

keep_awake() holds a request against idle sleep for as long as its `with` block
runs, in the thread that runs it (the app's session thread), and drops it when the
block ends, also on a Stop or an error. It changes no system setting: the display
may still turn off, and closing a laptop's lid still puts it to sleep.

- Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), which
  belongs to the calling thread (Windows also drops it when the thread ends);
  `powercfg /requests` lists it under SYSTEM.
- macOS: a power assertion of type PreventUserIdleSystemSleep (IOKit's
  IOPMAssertionCreateWithName, as `caffeinate -i` makes); `pmset -g assertions`
  lists it with REASON.
- Elsewhere: nothing.

If the request cannot be made, the block runs anyway (the computer may then
sleep as it would have before). keep_awake yields True if the request is held,
False if it could not be made, None where there is none to make.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from collections.abc import Iterator

REASON = "Clausa is analysing recordings"


@contextlib.contextmanager
def keep_awake(reason: str = REASON) -> Iterator[bool | None]:
    supported = os.name == "nt" or sys.platform == "darwin"
    release = None
    try:
        if os.name == "nt":
            release = _windows()
        elif sys.platform == "darwin":
            release = _macos(reason)
    except (OSError, AttributeError, ValueError):
        release = None
    try:
        yield (release is not None) if supported else None
    finally:
        if release is not None:
            try:
                release()
            except (OSError, AttributeError, ValueError):
                pass


def _windows():
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    if not kernel32.SetThreadExecutionState(es_continuous | es_system_required):
        return None
    return lambda: kernel32.SetThreadExecutionState(es_continuous)


def _macos(reason: str):
    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    iokit.IOPMAssertionCreateWithName.restype = ctypes.c_int32
    iokit.IOPMAssertionCreateWithName.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_uint32)]
    iokit.IOPMAssertionRelease.restype = ctypes.c_int32
    iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]
    utf8 = 0x08000100  # kCFStringEncodingUTF8
    kind = cf.CFStringCreateWithCString(None, b"PreventUserIdleSystemSleep", utf8)
    name = cf.CFStringCreateWithCString(None, reason.encode("utf-8"), utf8)
    assertion = ctypes.c_uint32(0)
    try:
        result = iokit.IOPMAssertionCreateWithName(kind, 255, name, ctypes.byref(assertion))  # kIOPMAssertionLevelOn
    finally:
        cf.CFRelease(kind)
        cf.CFRelease(name)
    if result != 0:
        return None
    return lambda: iokit.IOPMAssertionRelease(assertion.value)
