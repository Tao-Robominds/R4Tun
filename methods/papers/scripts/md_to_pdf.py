"""Convert a markdown file to a styled PDF via markdown → HTML → weasyprint.

Usage:
    python md_to_pdf.py                  # converts r4tun.md → r4tun.pdf
    python md_to_pdf.py r4tun_concise    # converts r4tun_concise.md → r4tun_concise.pdf
"""

import pathlib
import sys
import markdown
from weasyprint import HTML

ROOT = pathlib.Path(__file__).resolve().parent.parent
stem = sys.argv[1] if len(sys.argv) > 1 else "r4tun"
MD_PATH = ROOT / f"{stem}.md"
PDF_PATH = ROOT / f"{stem}.pdf"

CSS = """
@page {
    size: A4;
    margin: 25mm 20mm 25mm 20mm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #555; }
}
body {
    font-family: "Times New Roman", "DejaVu Serif", serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #111;
    max-width: 100%;
}
h1 { font-size: 16pt; text-align: center; margin-top: 0; }
h2 { font-size: 13pt; margin-top: 1.2em; border-bottom: 0.5pt solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 1em; }
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9pt;
    margin: 0.8em 0;
    page-break-inside: avoid;
}
th, td { border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left; }
th { background: #f0f0f0; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
code {
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 1pt 3pt;
    border-radius: 2pt;
}
pre {
    background: #f5f5f5;
    padding: 6pt 10pt;
    font-size: 9pt;
    overflow-x: auto;
    border: 0.5pt solid #ddd;
    border-radius: 3pt;
}
blockquote {
    border-left: 3pt solid #ccc;
    margin-left: 0;
    padding-left: 10pt;
    color: #555;
}
hr { border: none; border-top: 0.5pt solid #ccc; margin: 1em 0; }
strong { font-weight: bold; }
em { font-style: italic; }
p { margin: 0.4em 0; }
ul, ol { margin: 0.3em 0; padding-left: 20pt; }
"""


def convert():
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "smarty"],
    )
    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{CSS}</style>
</head><body>{html_body}</body></html>"""

    HTML(string=full_html).write_pdf(str(PDF_PATH))
    print(f"PDF written to {PDF_PATH}  ({PDF_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    convert()
