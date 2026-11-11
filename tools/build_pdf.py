"""Build paper.pdf from paper.md.

Markdown -> HTML (python-markdown + pymdown arithmatex), formulas rendered by
KaTeX, the source of abtest_sim/ inserted as Appendix B, then printed to PDF by
headless Google Chrome / Chromium.

Run: python tools/build_pdf.py
Needs: pip install markdown pymdown-extensions; Chrome or Chromium on PATH;
internet access to load KaTeX from cdn.jsdelivr.net.
"""
from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
CODE_FILES = ["config.py", "data.py", "corrupt.py", "stats.py", "simulate.py"]
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm; }
body { font-family: "Times New Roman", "Liberation Serif", "DejaVu Serif", serif;
       font-size: 11pt; line-height: 1.42; color: #111; background: #fff; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 6pt 0; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; border-bottom: 0.6pt solid #999; padding-bottom: 2pt; }
h3 { font-size: 12pt; margin: 12pt 0 4pt 0; }
p { margin: 0 0 7pt 0; text-align: justify; hyphens: auto; }
a { color: #1a4f8b; text-decoration: none; word-break: break-all; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; font-size: 8.6pt; width: 100%; }
th, td { border: 0.5pt solid #aaa; padding: 2pt 4pt; vertical-align: top; }
th { background: #eef1f5; }
td:not(:first-child) { white-space: nowrap; }
img { display: block; max-width: 100%; margin: 8pt auto 2pt auto; }
p.cap { font-size: 9pt; color: #333; text-align: center; margin-bottom: 12pt; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 8.8pt; }
pre { background: #f6f7f9; border: 0.5pt solid #ddd; padding: 5pt 7pt; font-size: 7.6pt;
      line-height: 1.3; white-space: pre-wrap; word-break: break-word; }
pre code { font-size: 7.6pt; }
li { margin-bottom: 3pt; }
h2, h3 { page-break-after: avoid; }
img, table, pre.short { page-break-inside: avoid; }
.katex { font-size: 1.02em; }
"""


def code_appendix() -> str:
    parts = []
    for name in CODE_FILES:
        src = (ROOT / "abtest_sim" / name).read_text(encoding="utf-8").rstrip()
        parts.append(f"### abtest_sim/{name}\n\n```python\n{src}\n```\n")
    return "\n".join(parts)


def figures_with_captions(text: str) -> str:
    """![caption](path) -> image followed by a caption paragraph."""
    import re
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
                  lambda m: f'<img src="{m.group(2)}" alt="">\n\n<p class="cap">{html.escape(m.group(1))}</p>',
                  text)


def main():
    md_text = (ROOT / "paper.md").read_text(encoding="utf-8")
    md_text = md_text.replace("<!-- CODE_APPENDIX -->", code_appendix())
    md_text = figures_with_captions(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<base href="{ROOT.as_uri()}/">
<title>Missing Values and Outliers in A/B Tests</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script src="{KATEX}/contrib/auto-render.min.js"></script>
<style>{CSS}</style></head><body>
{body}
<script>
renderMathInElement(document.body, {{
  delimiters: [{{left: "\\\\[", right: "\\\\]", display: true}},
               {{left: "\\\\(", right: "\\\\)", display: false}}],
  throwOnError: false }});
</script></body></html>"""
    BUILD.mkdir(exist_ok=True)
    out_html = BUILD / "paper.html"
    out_html.write_text(page, encoding="utf-8")

    chrome = next((c for c in ["google-chrome", "chromium", "chromium-browser", "chrome"]
                   if shutil.which(c)), None)
    if chrome is None:
        sys.exit(f"Chrome/Chromium not found. HTML is at {out_html}; print it to PDF manually.")
    pdf = ROOT / "paper.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={pdf}", out_html.as_uri()],
                   check=True, capture_output=True)
    print(f"written: {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
