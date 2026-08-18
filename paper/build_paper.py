"""Build the manuscript PDF from its Markdown source.

    python paper/build_paper.py

Markdown is the authoring format; the PDF is a build artefact that is committed so the
record is readable without a toolchain. Pandoc converts to LaTeX and XeLaTeX typesets it
-- XeLaTeX rather than pdfLaTeX because the text uses Greek and mathematical characters
directly (rho, times, minus, en/em dashes) and pdfLaTeX cannot set them without escaping
every one.

Requires pandoc and a TeX distribution on PATH. Neither is needed to use the repository:
the committed PDF and the figures are the deliverable, and every figure regenerates from
`results/*.json` with `src/make_figures.py` alone.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "stored_not_integrated.md"
PDF = HERE / "stored_not_integrated.pdf"
HEADER = HERE / "_preamble.tex"

TITLE = "Stored, Not Integrated: A Pre-Treatment Predictor and Three Controls for Knowledge Injection"
AUTHOR = "Alex Liu"
# Taken from the manuscript rather than hardcoded: the two drifted apart once, and a PDF
# dated differently from its own source is a defect a reader cannot diagnose.
def _date_from_manuscript() -> str:
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", SRC.read_text(encoding="utf-8")[:2000])
    if not m:
        raise SystemExit("no date found in the manuscript header")
    return m.group(1)


DATE = _date_from_manuscript()


def _figures_are_current() -> bool:
    """Check the figures against the data they were generated from.

    `make_figures.py` records the sha256 of every result file it read. Comparing those
    against the files on disk answers the question directly, and unlike a timestamp
    comparison it is unaffected by file-copy order -- in a freshly copied tree every
    mtime is identical, which made an mtime check fire on every figure.
    """
    import hashlib
    import json as _json

    lock = HERE / "figures" / "figure_inputs.lock.json"
    if not lock.exists():
        print("warning: no figure_inputs.lock.json; cannot verify the figures are current")
        print("  run `python src/make_figures.py` to generate the figures and the lock")
        return False
    recorded = _json.loads(lock.read_text(encoding="utf-8"))["inputs"]
    changed = []
    for rel, sha in recorded.items():
        p = Path(rel)
        if not p.exists():
            changed.append(rel + " (missing)")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            changed.append(rel)
    if changed:
        print(f"warning: {len(changed)} figure input(s) changed since the figures were built:")
        for c in changed[:6]:
            print(f"    {c}")
        print("  run `python src/make_figures.py` before building.")
        return False
    return True


def main() -> int:
    for tool in ("pandoc", "xelatex"):
        if shutil.which(tool) is None:
            print(f"error: {tool} not found on PATH.")
            print("  pandoc:  https://pandoc.org/installing.html")
            print("  xelatex: install TeX Live, MiKTeX or MacTeX")
            return 1
    if not _figures_are_current() and "--allow-stale-figures" not in sys.argv:
        print("refusing to build a PDF from stale figures; pass --allow-stale-figures to override")
        return 1
    if not SRC.exists():
        print(f"error: {SRC} not found")
        return 1

    # The Markdown opens with an H1 title so it reads correctly on GitHub. In the PDF the
    # title comes from the metadata block instead, so leaving the H1 in produces the title
    # twice and pushes every real section down a level -- "1.2 Introduction" rather than
    # "1 Introduction". Strip it for the build and promote everything back up one level.
    body = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    body = re.sub(r"\A#\s+[^\n]*\n", "", body)
    # The front-matter block (author, licence, keywords, cite-as) reads correctly on
    # GitHub, but pandoc places the table of contents immediately after the title block,
    # so in the PDF this lands awkwardly between the contents and the abstract. Its
    # content is already carried by the title block and the subtitle.
    body = re.sub(r"\A\s*\*\*Preprint.*?\n---\n", "", body, flags=re.S)
    tmp = HERE / "_build.md"
    tmp.write_text(body, encoding="utf-8")

    cmd = [
        "pandoc", str(tmp), "-o", str(PDF),
        "--from", "markdown+pipe_tables+yaml_metadata_block-raw_html",
        "--pdf-engine", "xelatex",
        "--metadata", f"title={TITLE}",
        "--metadata", f"author={AUTHOR}",
        "--metadata", f"date={DATE}",
        "--metadata", "subtitle=Preprint - CC BY 4.0 - archived on Zenodo",
        "--shift-heading-level-by=-1",
        "--toc", "--toc-depth=2",
        # figures are referenced relative to the markdown, not the working directory
        "--resource-path", str(HERE),
        "-V", "documentclass=article",
        "-V", "papersize=a4",
        "-V", "geometry:margin=2.4cm",
        "-V", "fontsize=10pt",
        "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "toccolor=black",
        "-V", "colorlinks=true",
        # a font with the Greek and dashes the text uses; fall back silently if absent
        "-V", "mainfont=Times New Roman",
        "-V", "monofont=Consolas",
        # keep figures near their text and never wider than the type block
        "-V", "graphics=true",
        "--highlight-style", "tango",
        # LaTeX floats figures to wherever it finds room, which put Figure 5 above its
        # own section heading and pushed a wide figure into the page furniture. Pin every
        # figure where it appears in the source, and never let one exceed the type block.
        "-H", str(HEADER),
    ]
    print("building", PDF.name)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Missing fonts are the usual cause; retry with pandoc's defaults before failing.
        print("  first attempt failed, retrying with default fonts")
        cmd = [c for c in cmd if not c.startswith(("mainfont=", "monofont="))]
        cmd = [c for i, c in enumerate(cmd)
               if not (c == "-V" and i + 1 < len(cmd) and cmd[i + 1].startswith(("mainfont", "monofont")))]
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return r.returncode

    tmp.unlink(missing_ok=True)
    size = PDF.stat().st_size / 1e6
    print(f"  wrote {PDF}  ({size:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
