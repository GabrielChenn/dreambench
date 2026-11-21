"""Shared inline theme for the workbench and portable replay files."""

WORKBENCH_CSS = """
:root{color-scheme:dark;--surface:#141d2a;--line:#29384b;--ink:#f2f6fc;--muted:#a9b8cc;--accent:#78efca}
*{box-sizing:border-box}
body{margin:0;background:#0c131e;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
header{border-bottom:1px solid var(--line);padding:1rem max(1.3rem, calc((100% - 76rem)/2));font-weight:750}
header span{color:var(--accent);margin-right:.55rem}
main{max-width:76rem;margin:auto;padding:2.5rem 1.3rem 4rem}
h1,h2,p{margin:0}h1{font-size:clamp(2.35rem,5vw,4rem);line-height:1.08;letter-spacing:-.05em}
h2{font-size:1.2rem;margin-bottom:.75rem}
.description,.detail{color:var(--muted);margin-top:.65rem}
a{color:var(--accent)}:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
"""
