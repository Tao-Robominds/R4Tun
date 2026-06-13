"""Render Response_to_Reviewers.md to PDF with response paragraphs in green."""

import re
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).parent
SRC = HERE / "Response_to_Reviewers.md"
OUT = HERE / "Response_to_Reviewers.pdf"


def wrap_responses(md_text: str) -> str:
    """Wrap each Response: block in a fenced div so CSS can colour it green.

    A response block starts at a line beginning with **Response:** and ends
    at the next reviewer/comment/section delimiter.
    """
    lines = md_text.splitlines()
    out, in_resp = [], False

    delimiter_re = re.compile(
        r"^\s*(\*\*Reviewer's comment|\*\*Comment\b|\*\*Q\d|\*\*General comment|---|##\s|#\s)"
    )

    for line in lines:
        starts = line.lstrip().startswith("**Response:**")
        if in_resp and (delimiter_re.match(line) or starts):
            out.append('</div>')
            in_resp = False
        if starts:
            out.append('<div class="response" markdown="1">')
            in_resp = True
        out.append(line)

    if in_resp:
        out.append('</div>')

    return "\n".join(out)


CSS = """
@page { size: A4; margin: 22mm 20mm; }
body {
    font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif;
    font-size: 10.5pt; line-height: 1.45; color: #111;
}
h1 { font-size: 18pt; margin: 0 0 12pt 0; }
h2 { font-size: 14pt; margin: 22pt 0 8pt 0; border-bottom: 1px solid #999;
     padding-bottom: 3pt; }
hr { border: 0; border-top: 1px dashed #bbb; margin: 18pt 0; }
p, li { margin: 4pt 0; }
ul, ol { margin: 4pt 0 4pt 18pt; padding: 0; }
code { background: #f3f3f3; padding: 0 3pt; border-radius: 2pt;
       font-family: "DejaVu Sans Mono", monospace; font-size: 9.5pt; }
blockquote { margin: 6pt 0 6pt 12pt; padding-left: 10pt;
             border-left: 3pt solid #bbb; color: #333; }

.response {
    color: #1b6e2b;
}
.response strong { color: #14591f; }
.response em { color: #1b6e2b; }
.response code { color: #14591f; background: #eaf4ec; }
.response blockquote { color: #1b6e2b; border-left-color: #4ea060; }
"""


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    wrapped = wrap_responses(md_text)
    html_body = markdown.markdown(
        wrapped,
        extensions=["extra", "sane_lists", "md_in_html"],
    )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    HTML(string=html_doc, base_url=str(HERE)).write_pdf(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
