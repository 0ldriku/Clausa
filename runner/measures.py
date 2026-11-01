"""Compute the nine measures again from annotated TextGrids, as they are now (the app's Compute, `./clausa measures`).

A run's measures come from its clauses TextGrids (clauses/<id>.TextGrid) and the
speech found inside pauses (islands/per_file/), through the language's calculator,
the run's last stage (runner/run.py). People correct the TextGrids in Praat; this
runs the same calculator, the same way, on the TextGrids as they are now. A
TextGrid nobody changed gives its run's row byte for byte: a recording's measures
do not depend on the other recordings. The same calculator run gives the words,
pauses and clauses the measures came from, as three tables beside the CSV
(pipeline/shared/uf_tables.py; projects.table_path names them).

Before the calculator, each TextGrid is checked (textgrid_problem): it must be
there, be a text TextGrid (praatio reads UTF-8 and UTF-16, not Praat's binary
format), and have a words tier and a clauses tier under the names the calculators
look for. The calculator would otherwise carry on without them (the first tier as
words, no clauses) and give wrong measures.
"""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable

from runner import envs
from runner.projects import TABLES, read_csv_rows, read_table_rows, table_path, write_files
from runner.run import LANGUAGES, MODES, _command, _run_command

# The tier names the calculators accept (pipeline/*/caf_calculator_*.py), words first.
TIERS = {
    "words": {"en": ("words", "word", "Word", "Words"), "ja": ("words", "word", "Word", "Words", "単語", "transcription")},
    "clauses": {"en": ("clauses", "clause", "Clauses", "Clause"), "ja": ("clauses", "clause", "Clauses", "Clause")},
}
NO_ROW = "The measures could not be computed. The log has the details."
_LONG_NAME = re.compile(r'^\s*name\s*=\s*"((?:[^"]|"")*)"\s*$', re.M)
_SHORT_NAME = re.compile(r'^\s*"(?:IntervalTier|TextTier)"\s*\n\s*"((?:[^"]|"")*)"\s*$', re.M)


class Problem(Exception):
    """Why a recording has no measures: code (compute.err.<code> in the window's strings), English text, params."""

    def __init__(self, code: str, text: str, **params):
        super().__init__(text)
        self.code, self.text, self.params = code, text, params


def tier_names(data: bytes) -> list[str]:
    """The tier names of a text TextGrid (long or short format, UTF-8 or UTF-16), as Praat writes them."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")
    text = text.replace("\r\n", "\n")
    names = _LONG_NAME.findall(text) or _SHORT_NAME.findall(text)  # long format (name = "..."), else short
    return [n.replace('""', '"') for n in names]


def textgrid_problem(path: Path, language: str) -> Problem | None:
    """Why the measures cannot be computed from this TextGrid, or None."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return Problem("missing", f"Its TextGrid is missing: {path.name}", file=path.name)
    except OSError as exc:
        return Problem("unreadable", f"Its TextGrid could not be read ({exc}). In Praat, save it as a text file.",
                       error=str(exc))
    if data.startswith(b"ooBinaryFile"):
        return Problem("binary", "Its TextGrid is saved in Praat's binary format. In Praat, save it as a text file.")
    try:
        names = tier_names(data)
    except UnicodeDecodeError:
        return Problem("not_text", "Its TextGrid could not be read as text. In Praat, save it as a text file.")
    if not names:
        return Problem("no_tiers", "Its TextGrid has no tiers. In Praat, save it as a text file.")
    for tier in ("words", "clauses"):
        if not set(names) & set(TIERS[tier][language]):
            return Problem("no_tier", f"Its TextGrid has no {tier} tier. Add it again in Praat, named {tier}.",
                           tier=tier)
    return None


def _rows(csv_path: Path) -> tuple[bytes, dict[str, bytes], dict[str, str]]:
    """The calculator's CSV: (header, {id: line} of the rows without an error, {id: error})."""
    header, lines = read_csv_rows(csv_path)
    errors = {}
    names = next(csv.reader(io.StringIO(header.decode("utf-8"))), []) if header else []
    if "error" in names:
        col = names.index("error")
        for rec_id, line in list(lines.items()):
            values = next(csv.reader(io.StringIO(line.decode("utf-8"))), [])
            if col < len(values) and values[col].strip():
                errors[rec_id] = values[col].strip()
                del lines[rec_id]
    return header, lines, errors


def compute(language: str, groups: list[tuple[Path, list[str]]], output: Path, *,
            stop: threading.Event | None = None,
            on_file: Callable[[str, Problem | None], None] | None = None,
            order: list[str] | None = None, paper_compat: bool = False) -> dict[str, Problem | None]:
    """Compute the measures of these recordings from their TextGrids; write them to output.

    groups: (run folder, ids) pairs; the TextGrids are <run folder>/clauses/<id>.TextGrid,
    the speech inside pauses <run folder>/islands/per_file/. output: a CSV like a run's
    uf_measures.csv, one row per recording with measures, in the order given; written
    when all are computed, removed if none has measures; the same for the tables
    beside it (table_path), with these recordings' rows. stop: as
    run_pipeline's (Stopped is raised and output is left as it was). on_file(id, problem)
    as each recording ends. order: the rows' order (default: as in groups). paper_compat:
    paper mode's calculator, for a run made with --paper-compat (no tables). The four files
    are replaced together (projects.write_files). Returns {id: None, or the Problem}.
    """
    lang, mode = LANGUAGES[language], MODES[paper_compat]
    script = mode.pipeline_dir / language / lang.uf_script
    env = envs.stage_env(lang.uf_env, pipeline_dir=mode.pipeline_dir, aliases=mode.env_aliases)
    results: dict[str, Problem | None] = {}
    lines: dict[str, bytes] = {}
    header = None
    table_headers: dict[str, bytes] = {}
    table_lines: dict[str, dict[str, list[bytes]]] = {t: {} for t in TABLES}

    def ended(rec_id: str, problem: Problem | None) -> None:
        results[rec_id] = problem
        if on_file is not None:
            on_file(rec_id, problem)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compute-", dir=output.parent) as tmp:
        for n, (run_dir, ids) in enumerate(groups):
            ready = []
            for rec_id in ids:
                problem = textgrid_problem(run_dir / "clauses" / f"{rec_id}.TextGrid", language)
                if problem is not None:
                    ended(rec_id, problem)
                else:
                    ready.append(rec_id)
            # A file the calculator cannot read gives a row with an error column, and the CSV
            # then has that column for every row: once one failed, the others are computed again.
            for attempt in (1, 2):
                if not ready:
                    break
                file_list, out_csv = Path(tmp) / f"{n}-{attempt}.json", Path(tmp) / f"{n}-{attempt}.csv"
                tables_dir = Path(tmp) / f"{n}-{attempt}"
                file_list.write_text(json.dumps({"files": ready}, ensure_ascii=False), encoding="utf-8")
                _, cmd = _command(lang.uf_env, script, run_dir / "clauses",
                                  "--candidate-dir", run_dir / "islands" / "per_file",
                                  "--file-list-json", file_list, "--file-list-key", "files", "--output", out_csv,
                                  *(["--tables-dir", tables_dir] if mode.tables else []))
                _run_command(cmd, env, stop=stop)
                h, found, errors = _rows(out_csv) if out_csv.exists() else (b"", {}, {})
                tables = {}
                for t in TABLES:
                    path = table_path(tables_dir / "uf_measures.csv", t)
                    if path.exists():
                        th, rows = read_table_rows(path)
                        grouped: dict[str, list[bytes]] = {}
                        for rec_id, line in rows:
                            grouped.setdefault(rec_id, []).append(line)
                        tables[t] = (th, grouped)
                for rec_id in ready:
                    if rec_id in errors:
                        ended(rec_id, Problem("calculator", f"The measures could not be computed ({errors[rec_id]}). "
                                              "The log has the details.", error=errors[rec_id]))
                    elif rec_id not in found:
                        ended(rec_id, Problem("no_row", NO_ROW))
                if errors:
                    ready = [i for i in ready if i in found]
                    continue
                for rec_id in ready:
                    if header is None:
                        header = h
                    if h == header:
                        lines[rec_id] = found[rec_id]
                        for t, (th, grouped) in tables.items():
                            table_headers.setdefault(t, th)
                            table_lines[t][rec_id] = grouped.get(rec_id, [])
                        ended(rec_id, None)
                    else:  # not expected: one calculator writes one set of columns
                        ended(rec_id, Problem("other_columns", "Its measures came with other columns. The log has "
                                                               "the details."))
                break
            for rec_id in ids:  # failed again on the second attempt
                if rec_id not in results:
                    ended(rec_id, Problem("no_row", NO_ROW))
    given = order if order is not None else [rec_id for _, ids in groups for rec_id in ids]
    order = [rec_id for rec_id in given if rec_id in lines]
    files = {table_path(output, t): (table_headers[t] + b"".join(b"".join(table_lines[t].get(i, [])) for i in order)
                                     if order and t in table_headers else None) for t in TABLES}
    files[output] = header + b"".join(lines[i] for i in order) if order else None
    write_files(files)  # all four, or (a file open in another program) none
    return results


def main(language: str, run_dir: Path, output: Path | None = None) -> int:
    """`./clausa measures`: the measures of every TextGrid in <run_dir>/clauses, as they are now."""
    clauses = run_dir / "clauses"
    if not clauses.is_dir():
        print(f"ERROR: {clauses} not found: give the output folder of `./clausa run`")
        return 2
    ids = sorted(p.stem for p in clauses.glob("*.TextGrid") if not p.name.startswith("."))  # not macOS's ._ files
    if not ids:
        print(f"ERROR: no TextGrid in {clauses}")
        return 2
    output = output or run_dir / "measures.csv"
    try:
        paper = bool(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("paper_compat"))
    except (OSError, ValueError, AttributeError):
        paper = False
    if paper:
        print("This run was made with --paper-compat: paper mode's calculator computes the measures (no tables).")
    try:
        results = compute(language, [(run_dir, ids)], output, paper_compat=paper)
    except OSError as exc:
        print(f"ERROR: {exc}")
        return 1
    failed = {i: p for i, p in results.items() if p is not None}
    for rec_id, problem in failed.items():
        print(f"{rec_id}: {problem.text}")
    done = len(results) - len(failed)
    print(f"\n{done} of {len(results)} computed" + (f": {output}" if done else ""))
    if done and not paper:
        print("Their words, pauses and clauses: " + ", ".join(table_path(output, t).name for t in TABLES))
    return 1 if failed else 0
