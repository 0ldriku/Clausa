#!/usr/bin/env python3
"""The analysis repository's Japanese reference calculator with a tool calculator's mora count.

  ja_reference_measures.py <tool calculator .py> <reference_textgrids> --output <csv>

Run from the analysis repository (as run_rq_variant.sh does). The reference
measures then count morae the same way as the automatic side scored with
CALC_REPO, also when the tool's count differs from the analysis repository's.
The tool's calculator reads with SudachiPy, so this needs an environment that
has it (the analysis environment, or ja-electra310).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tool_calculator = Path(sys.argv.pop(1))
    # By path: the analysis repository has a frozen file of the same name as the tool's calculator.
    reference = _load("caf_calculator_ja", Path("code/ja/caf_calculator_ja.py"))
    tool = _load("tool_caf_calculator_ja", tool_calculator)
    extract_tiers = reference.TextGridReaderJA._extract_tiers

    def extract_tiers_tool_mora(self):
        extract_tiers(self)
        words = [w for w in self.words if not w.is_pause and w.text.lower() not in self.FILLER_MARKERS]
        texts = [w.text for w in words]
        if hasattr(tool, "mora_counts"):
            counts = tool.mora_counts(texts)
        else:  # a calculator from before numbers were read by place
            counts = [tool.count_mora(r) for r in tool.kana_readings(texts)]
        for word, morae in zip(words, counts):
            word.mora_count = morae
        self.total_mora = sum(w.mora_count for w in words)

    reference.TextGridReaderJA._extract_tiers = extract_tiers_tool_mora
    reference.main()


if __name__ == "__main__":
    main()
