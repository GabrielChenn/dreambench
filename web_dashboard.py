"""Serve DreamBench experiment reports and replay files on loopback."""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent / "outputs"
ARTIFACTS = {"replay.html", "replay-yolo.html", "replay-detections.html", "video.mp4"}
OUTCOMES = {"success": "Passed", "task-failure": "Failed", "invalid-run": "Invalid run",
            "unsupported-task": "Unsupported task", "insufficient-evidence": "Insufficient evidence"}
DECISIONS = {"keep-baseline": "Keep baseline", "retest-candidate": "Retest candidate",
             "insufficient-evidence": "Insufficient evidence"}

CSS = """
:root{color-scheme:dark;--surface:#141d2a;--line:#29384b;--ink:#f2f6fc;--muted:#a9b8cc;--accent:#78efca}
*{box-sizing:border-box}
body{margin:0;background:#0c131e;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
header{border-bottom:1px solid var(--line);padding:1rem max(1.3rem, calc((100% - 76rem)/2));font-weight:750}
header span{color:var(--accent);margin-right:.55rem}
main{max-width:76rem;margin:auto;padding:2.5rem 1.3rem 4rem}
h1,h2,p{margin:0}h1{font-size:clamp(2.35rem,5vw,4rem);line-height:1.08;letter-spacing:-.05em}
h2{font-size:1.2rem;margin-bottom:.75rem}
.description,.detail{color:var(--muted);margin-top:.65rem}
.layout{display:grid;grid-template-columns:17rem 1fr;gap:1.35rem;margin-top:2rem}
aside,article,.decision{background:var(--surface);border:1px solid var(--line);border-radius:.9rem;padding:1.2rem}
aside{align-self:start}aside ul{padding:0;margin:0;list-style:none}
aside a{display:block;border-radius:.55rem;padding:.65rem .7rem;margin-bottom:.25rem;text-decoration:none}
aside a[aria-current=page]{background:#23354a;color:var(--accent)}
a{color:var(--accent)}a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
aside small{display:block;color:var(--muted);font-size:.85rem}
.decision{border-left:4px solid var(--accent);margin-bottom:1rem}
.decision h2{font-size:1.45rem;margin:0}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.head{display:flex;gap:.6rem;align-items:center;justify-content:space-between;flex-wrap:wrap}
.head h2{margin:0}strong.pill{font-size:.88rem;background:#293449;border-radius:2rem;padding:.22rem .65rem}
strong.success{background:#16463a;color:#adf6dc}strong.task-failure{background:#4a2933;color:#ffc5d0}
dl{margin:1rem 0;display:grid;gap:.5rem}dl div{display:flex;justify-content:space-between;gap:.75rem;border-bottom:1px solid var(--line);padding-bottom:.45rem}
dt{color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums;text-align:right}
.link{display:inline-block;margin-top:1rem;font-weight:650}
@media(max-width:820px){.layout{grid-template-columns:1fr}}
@media(max-width:540px){.cards{grid-template-columns:1fr}}
"""


def safe_artifact(root: Path, name: str) -> Path | None:
    if "\x00" in name:
        return None
    path = (root / name).resolve()
    if path.is_relative_to(root) and path.is_file() and path.name in ARTIFACTS:
        return path
    return None


def replay_url(root: Path, report: Path, recorded: str | None, actions: object, label: str) -> str | None:
    if isinstance(recorded, str) and recorded:
        path = Path(recorded)
        if path.is_absolute() and path.is_relative_to(root):
            candidate = safe_artifact(root, str(path.relative_to(root)))
            if candidate:
                return "/artifacts/" + quote(candidate.relative_to(root).as_posix())
        if path.name in ARTIFACTS and path.parent.name.startswith("replay-") and isinstance(actions, int):
            # A report copied from a teammate may contain their absolute path.
            relocated = report.parent / f"actions-{actions}" / path.parent.name / path.name
            for location in (report.parent / label.lower() / path.parent.name / path.name, relocated):
                candidate = safe_artifact(root, str(location.relative_to(root)))
                if candidate:
                    return "/artifacts/" + quote(candidate.relative_to(root).as_posix())
    return None


def experiments(root: Path) -> list[tuple[Path, dict]]:
    result = []
    reports = [path for path in root.glob("comparison-*/comparison.json")
               if path.resolve().is_relative_to(root)]
    for path in sorted(reports, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(data, dict) and all(isinstance(data.get(k), dict) for k in ("baseline", "candidate")):
            result.append((path, data))
    return result


def value(item: object, suffix: str = "") -> str:
    if item is None:
        return "—"
    return escape(f"{item:.2f}" if isinstance(item, float) else str(item)) + suffix


def trial_card(root: Path, path: Path, label: str, trial: dict) -> str:
    outcome = str(trial.get("outcome", "Unknown"))
    url = replay_url(root, path, trial.get("replay"), trial.get("actions"), label)
    metrics = (
        ("Action limit", value(trial.get("actions"), " steps")),
        ("API requests", value(trial.get("reported_api_requests"))),
        ("Wall time", value(trial.get("wall_elapsed_s"), " s")),
        ("Evidence time", value(trial.get("evidence_sim_s"), " s sim")),
    )
    rows = "".join(f"<div><dt>{key}</dt><dd>{metric}</dd></div>" for key, metric in metrics)
    link = f'<a class="link" href="{escape(url)}">Open {label.lower()} replay</a>' if url else ""
    return (
        f'<article><div class="head"><h2>{label}</h2>'
        f'<strong class="pill {escape(outcome)}">{escape(OUTCOMES.get(outcome, outcome))}</strong></div>'
        f'<p class="detail">{value(trial.get("prompt", trial.get("task")))}</p>'
        f'<dl>{rows}</dl><p class="detail">{value(trial.get("reason"))}</p>{link}</article>'
    )


def dashboard(root: Path, selected: str | None) -> str:
    root = root.resolve()
    reports = experiments(root)
    entry = next((item for item in reports if item[0].parent.name == selected), None)
    if entry is None and selected is None and reports:
        entry = reports[0]
    links = []
    for path, report in reports:
        name = path.parent.name
        active = ' aria-current="page"' if entry and entry[0] == path else ""
        label = f'{value(report["baseline"].get("actions"))} vs {value(report["candidate"].get("actions"))} steps'
        if (report.get("experiment") or {}).get("comparison_variable") == "prompt":
            label = "Prompt comparison"
        links.append(f'<li><a href="/?run={quote(name)}"{active}>{label}<small>{escape(name)}</small></a></li>')
    sidebar = "".join(links) or "<li>No local comparisons</li>"
    if entry:
        path, report = entry
        baseline, candidate = report["baseline"], report["candidate"]
        decision = str(report.get("decision", "Unknown"))
        task = value(baseline.get("task_id") or baseline.get("task"))
        detail = f'{task} · model {value(baseline.get("model"))} · {value(baseline.get("seconds"), " s")} limit'
        content = (
            f'<div class="decision"><h2>{escape(DECISIONS.get(decision, decision))}</h2>'
            f'<p class="detail">{value(report.get("reason"))}</p><p class="detail">{detail}</p></div>'
            '<div class="cards">'
            + trial_card(root, path, "Baseline", baseline)
            + trial_card(root, path, "Candidate", candidate)
            + '</div>'
        )
    else:
        content = ('<article><h2>Comparison unavailable</h2><p class="detail">'
                   'Select a local report, or generate a comparison with <code>compare_replays.py</code>.'
                   '</p></article>')
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regression Workbench | DreamBench</title><style>{CSS}</style>
<header><span>◆</span>DreamBench &nbsp; <a href="/configure">Configure experiment</a></header><main>
<h1>Regression Workbench</h1>
<p class="description">Compare Dream-1 simulation runs and inspect the evidence behind each outcome.</p>
<div class="layout"><aside><nav aria-label="Experiments"><h2>Experiments</h2><ul>{sidebar}</ul></nav></aside>
<section aria-label="Comparison">{content}</section></div></main></html>"""


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path in ("/", "/configure"):
                if request.path == "/configure":
                    payload = Path(__file__).with_name("experiment_config.html").read_bytes()
                else:
                    selected = parse_qs(request.query).get("run", [None])[0]
                    payload = dashboard(root, selected).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            prefix = "/artifacts/"
            path = safe_artifact(root, unquote(request.path[len(prefix):])) if request.path.startswith(prefix) else None
            if path is None:
                self.send_error(404)
                return
            self.serve_artifact(path)

        def serve_artifact(self, path: Path) -> None:
            size = path.stat().st_size
            start, stop = 0, size - 1
            byte_range = self.headers.get("Range")
            if byte_range:
                limits = byte_range.removeprefix("bytes=").split("-", 1)
                if (not byte_range.startswith("bytes=") or len(limits) != 2
                        or not all(not n or (n.isascii() and n.isdecimal() and len(n) <= 20) for n in limits)
                        or not any(limits)):
                    self.send_error(416)
                    return
                low, high = limits
                start = int(low) if low else max(0, size - int(high))
                stop = min(size - 1, int(high)) if low and high else size - 1
                if start > stop:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = max(0, stop - start + 1)
            with path.open("rb") as handle:
                self.send_response(206 if byte_range else 200)
                self.send_header("Content-Type", "video/mp4" if path.suffix == ".mp4" else "text/html; charset=utf-8")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if byte_range:
                    self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
                self.end_headers()
                handle.seek(start)
                while length:
                    chunk = handle.read(min(length, 65536))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    length -= len(chunk)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--outputs", type=Path, default=ROOT)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.outputs.resolve())) as server:
        print(f"DreamBench · http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
