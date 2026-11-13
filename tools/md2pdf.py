"""Convert a Markdown report (with relative images) to PDF.

Usage:
    uv run --with pypandoc_binary --with typst submission/tools/md2pdf.py <report.md> <out.pdf>

Pipeline: Markdown --(pandoc)--> Typst --(typst)--> PDF. No LaTeX install needed.
Set the environment variable MD2PDF_IMAGE_WIDTH (e.g. "100%") to scale every image to that
fraction of the image block (useful for wide multi-panel figures); by default images keep
their natural size.
"""

import os
import sys
from pathlib import Path

import pypandoc
import typst

PREAMBLE = """#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#show raw.where(block: true): it => block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, text(size: 8.5pt, it))
#show table: set text(size: 9pt)
#show figure: set block(breakable: false)
#show image: it => align(center, block(width: 85%, it))
#let horizontalrule = line(length: 100%, stroke: 0.5pt + gray)
"""


def main() -> None:
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    body = pypandoc.convert_file(
        str(src),
        to="typst",
        format="markdown+tex_math_dollars+pipe_tables",
        extra_args=["--wrap=none"],
    )
    typ = src.with_suffix(".typ")
    preamble = PREAMBLE
    if os.environ.get("MD2PDF_IMAGE_WIDTH"):
        preamble += f"#set image(width: {os.environ['MD2PDF_IMAGE_WIDTH']})\n"
    typ.write_text(preamble + "\n" + body, encoding="utf-8")
    try:
        typst.compile(str(typ), output=str(out), root=str(src.parent))
    finally:
        typ.unlink(missing_ok=True)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
