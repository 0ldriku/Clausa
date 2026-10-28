#!/usr/bin/env python
"""
English Stage 2 with easyaligner: pre_mfa_json -> final JSON/TextGrid.

Drop-in alternative to mfa_from_pre_mfa_json.py; see shared/ctc_alignment.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents]:
    if (_p / "shared" / "ctc_alignment.py").exists():
        sys.path.insert(0, str(_p))
        break

import asr_mfa_en as pipeline
from shared.ctc_alignment import run_cli


def build_sequence(words, duration):
    return pipeline._build_augmented_sequence(words)[0]


if __name__ == "__main__":
    run_cli(pipeline, "en", build_sequence)
