"""Render paper/paper.md (with its figures) to paper/paper.pdf through headless Chrome.

    python paper/build_pdf.py [path-to-chrome]

Markdown is rendered in the page by marked (loaded from cdnjs), so the build needs a network connection.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = sys.argv[1] if len(sys.argv) > 1 else "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Can Creativity Be Reverse-Engineered?</title>
<base href="__BASE__">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=Literata:ital,opsz,wght@0,7..72,400..700;1,7..72,400&family=IBM+Plex+Mono:wght@400;500&display=block">
<style>
@page { size: A4; margin: 20mm 19mm 22mm; }
:root { --ink: #141821; --ink-2: #444b59; --rule: #c9ced8; --night: #1f3a64; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; color: var(--ink); font: 400 10pt/1.5 "Literata", Georgia, serif; hyphens: auto; }
h1, h2, h3 { font-family: "Archivo", "Helvetica Neue", Arial, sans-serif; font-stretch: 80%; line-height: 1.15; break-after: avoid; text-wrap: balance; }
h1 { font-size: 21pt; font-weight: 850; margin: 0 0 6pt; }
.byline { font-family: "Archivo", Arial, sans-serif; font-stretch: 90%; font-size: 10.5pt; color: var(--ink-2); margin: 0 0 16pt; padding-bottom: 10pt; border-bottom: 1.5pt solid var(--ink); }
h2 { font-size: 13pt; font-weight: 800; margin: 16pt 0 5pt; }
h3 { font-size: 10.5pt; font-weight: 750; margin: 11pt 0 3pt; }
p { margin: 0 0 6pt; text-align: justify; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; }
li { margin-bottom: 2pt; }
a { color: var(--night); text-decoration: none; }
code { font-family: "IBM Plex Mono", Menlo, monospace; font-size: 8.4pt; }
.abstract { background: #eef1f6; padding: 9pt 12pt 4pt; margin: 0 0 8pt; border-radius: 3pt; }
.abstract h2 { margin-top: 0; }
figure, img { break-inside: avoid; }
p:has(> img), p:has(+ table) { break-after: avoid; } /* keep a figure with its caption, a table caption with its table */
p:has(> img) + p, p:has(+ table) { break-inside: avoid; } /* and never split a caption */
img { display: block; max-width: 100%; margin: 8pt auto 4pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.4pt; line-height: 1.35; margin: 4pt 0 10pt; break-inside: avoid; }
th, td { text-align: left; vertical-align: top; padding: 3pt 5pt; border-bottom: 0.6pt solid var(--rule); }
thead th { font-family: "Archivo", Arial, sans-serif; font-stretch: 88%; font-weight: 700; border-bottom: 1pt solid var(--ink); }
.refs ol { font-size: 8.6pt; line-height: 1.4; padding-left: 18pt; }
.refs li { text-align: left; }
.endmatter p { font-size: 8.8pt; }
</style></head>
<body>
<div id="out"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
<script>
const md = __MD__;
const lines = md.split("\\n");
const title = lines[0].replace(/^#\\s*/, "");
const byline = lines[2];
let body = lines.slice(3).join("\\n");
let html = marked.parse(body, { gfm: true });
const out = document.getElementById("out");
out.innerHTML = `<h1>${title}</h1><p class="byline">${byline}</p>` + html;
// wrap the abstract, the references and the end matter so they can be styled
const wrap = (startText, stopTag, cls) => {
  const h = [...out.querySelectorAll("h2")].find((x) => x.textContent.trim().startsWith(startText));
  if (!h) return;
  const box = document.createElement("section"); box.className = cls; h.before(box);
  let n = h; const nodes = [];
  while (n && !(n !== h && n.tagName === stopTag)) { nodes.push(n); n = n.nextElementSibling; }
  nodes.forEach((x) => box.appendChild(x));
};
wrap("Abstract", "H2", "abstract");
wrap("References", "H2", "refs");
wrap("Data, code", "H2", "endmatter");
document.fonts.ready.then(() => { document.title = "ready"; });
</script>
</body></html>
"""


def main():
    md = (HERE / "paper.md").read_text()
    html = PAGE.replace("__MD__", json.dumps(md)).replace("__BASE__", HERE.as_uri() + "/")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "paper.html"
        src.write_text(html)
        out = HERE / "paper.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                        "--virtual-time-budget=15000", f"--print-to-pdf={out}", src.as_uri()],
                       check=True, capture_output=True)
    print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
