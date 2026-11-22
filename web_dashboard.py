"""Serve DreamBench experiment reports and replay files on loopback."""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from compare_replays import validate_config
from dreamlabs.replay_page import replay_html
from dreamlabs.failure_modes import LABELS as MODE_LABELS, MODES, failure_mode, mode_distribution
from dreamlabs.perturbation import layout_from
from dreamlabs.safety import RULES as SAFETY_RULES, safety_summary
from dreamlabs.review_client import ReviewRenderer, ReviewUnavailable
from dreamlabs.scoring import verdict as judge_verdict
from dreamlabs.task_parser import GENERIC_NOTE
from dreamlabs.web_jobs import JobConflict, SimWorkspace
from dreamlabs.web_theme import WORKBENCH_CSS
from dreamlabs.yolo_status import read_detections, yolo_note, yolo_unavailable

ROOT = Path(__file__).resolve().parent / "outputs"
ARTIFACTS = {"replay.html", "replay-yolo.html", "replay-detections.html", "video.mp4"}
# Verdicts from dreambench_bk/judge_run.py; the outcome explains how the scorer got there.
OUTCOMES = {"success": "Placed", "task-failure": "Task failure", "invalid-run": "Invalid run",
            "run-error": "Run error", "unsupported-task": "Unsupported task",
            "insufficient-evidence": "Insufficient evidence", "not-run": "Not run",
            "invalid-start": "Invalid start"}
STOP_REASONS = {"finished": "Time limit", "home": "Arms back home", "operator": "Stopped from the page",
                "error": "Error", "interrupted": "Interrupted"}
DECISIONS = {"keep-baseline": "Keep baseline", "retest-candidate": "Retest candidate", "single-run": "Single run",
             "insufficient-evidence": "Insufficient evidence"}

CSS = """
.layout{display:grid;grid-template-columns:17rem 1fr;gap:1.35rem;margin-top:2rem}
aside,article,.decision{background:var(--surface);border:1px solid var(--line);border-radius:.9rem;padding:1.2rem}
aside{align-self:start}aside ul{padding:0;margin:0;list-style:none}
aside a{display:block;border-radius:.55rem;padding:.65rem .7rem;margin-bottom:.25rem;text-decoration:none}
aside a[aria-current=page]{background:#23354a;color:var(--accent)}
aside small{display:block;color:var(--muted);font-size:.85rem}
.decision{border-left:4px solid var(--accent);margin-bottom:1rem}
.decision h2{font-size:1.45rem;margin:0}
.scene{margin-top:1.55rem}.scene h3{margin:.2rem 0 .7rem}.cards{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.cards.single{grid-template-columns:minmax(0,40rem)}
.head{display:flex;gap:.6rem;align-items:center;justify-content:space-between;flex-wrap:wrap}
.head h2{margin:0}strong.pill{font-size:.88rem;background:#293449;border-radius:2rem;padding:.22rem .65rem}
strong.successed{background:#16463a;color:#adf6dc}
.generalization{margin-bottom:1rem}.generalization h2{margin:0 0 .3rem;font-size:1.2rem}
.generalization table{border-collapse:collapse;margin-top:.8rem;width:100%}
.generalization th,.generalization td{border-bottom:1px solid var(--line);padding:.45rem .6rem;text-align:left}
.generalization tfoot td{font-weight:600}.generalization a{text-decoration:none}
.generalization th small,.generalization td small{display:block;color:var(--muted);font-weight:400;font-size:.82rem}
.safety{display:inline-block;border-radius:.35rem;padding:0 .35rem;background:#293449;font-size:.85rem}
.safety.pass{background:#16463a;color:#adf6dc}.safety.review{background:#4a3d1c;color:#ffe1a3}
.safety.block{background:#4a2933;color:#ffc5d0}.muted{color:var(--muted)}tr.quiet td{color:var(--muted)}.scroll{overflow-x:auto}
.generalization td.num{font-variant-numeric:tabular-nums;white-space:nowrap}.generalization td.judged{color:var(--accent)}
.cell{display:inline-block;min-width:1.6rem;border-radius:.35rem;padding:.1rem .35rem;margin:.05rem;background:#293449}
.cell.successed{background:#16463a;color:#adf6dc}.cell.failed{background:#4a2933;color:#ffc5d0}strong.failed{background:#4a2933;color:#ffc5d0}
.what{margin-top:.8rem;padding:.6rem .75rem;border-radius:.5rem;background:#0f1824;color:#d3e5f6}
dl{margin:1rem 0;display:grid;gap:.5rem}dl div{display:flex;justify-content:space-between;gap:.75rem;border-bottom:1px solid var(--line);padding-bottom:.45rem}
dt{color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums;text-align:right}
.motion h3{margin:1rem 0 .5rem;font-size:1rem}.motion dl{margin:.5rem 0}
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
        if not isinstance(data, dict):
            continue
        pairs = data.get("scenes", [data])
        if isinstance(pairs, list) and pairs and all(
            isinstance(pair, dict) and isinstance(pair.get("baseline"), dict)
            and isinstance(pair.get("candidate", {}), dict)  # single runs have no candidate
            for pair in pairs
        ):
            result.append((path, data))
    return result


def value(item: object, suffix: str = "") -> str:
    if item is None:
        return "—"
    return escape(f"{item:.2f}" if isinstance(item, float) else str(item)) + suffix


def mapping(item: object) -> dict:
    """Treat a missing or malformed report section as empty."""
    return item if isinstance(item, dict) else {}


def is_number(item: object) -> bool:
    return type(item) in (int, float)


def trial_verdict(trial: dict) -> str | None:
    """The trial's verdict; older reports without one are judged from their outcome."""
    if trial.get("outcome") in ("not-run", "invalid-start"):
        return None
    stored = trial.get("verdict")
    if stored == "timeout stopped":  # older reports: a run that did not succeed is failed, however it ended
        return "failed"
    return stored if isinstance(stored, str) else judge_verdict(trial.get("outcome"), trial.get("stop_reason"))


def trial_card(root: Path, path: Path, label: str, trial: dict, folder: str | None = None) -> str:
    """One trial's result. folder is the trial's recording folder name when it differs from the label."""
    outcome = str(trial.get("outcome", "Unknown"))
    verdict = trial_verdict(trial)
    url = replay_url(root, path, trial.get("replay"), trial.get("actions"), folder or label)
    stop = trial.get("stop_reason")
    metrics = (
        ("Scorer result", escape(OUTCOMES.get(outcome, outcome))),
        ("Run ended by", escape(STOP_REASONS.get(stop, stop)) if isinstance(stop, str) else "—"),
        ("Action limit", value(trial.get("actions"), " steps")),
        ("API requests", value(trial.get("reported_api_requests"))),
        ("Wall time", value(trial.get("wall_elapsed_s"), " s")),
        ("Evidence time", value(trial.get("evidence_sim_s"), " s sim")),
        ("Failure mode", escape(MODE_LABELS[failure_mode(trial)])),
        ("Real-robot safety", safety_pill(trial)),
    )
    rows = "".join(f"<div><dt>{key}</dt><dd>{metric}</dd></div>" for key, metric in metrics)
    motion = trial.get("motion_quality")
    quality = ""
    if isinstance(motion, dict):
        quality_metrics = (
            ("Joint tracking RMSE", value(motion.get("tracking_rmse_rad"), " rad")),
            ("Joint jerk P95", value(motion.get("joint_jerk_p95_rad_s3"), " rad/s³")),
        )
        quality_rows = "".join(f"<div><dt>{key}</dt><dd>{metric}</dd></div>" for key, metric in quality_metrics)
        quality = (
            f'<div class="motion"><h3>Motion quality</h3><dl>{quality_rows}</dl>'
            '<p class="detail">Whole recording, including warm-up. Lower means less tracking error or jerk.</p></div>'
        )
    contacts = mapping(trial.get("contact_summary"))
    contact_rows = ""
    if contacts.get("sampled_frames"):
        events = mapping(contacts.get("events"))
        for category, name in (("grasp_contact", "Finger–fruit contacts"),
                               ("arm_table", "Arm–table collisions"), ("arm_arm", "Arm–arm collisions")):
            item = mapping(events.get(category))
            evidence = item.get("first_sim_s")
            timestamp = (f' · <a href="{escape(url)}?t={value(evidence)}">{value(evidence, " s sim")}</a>'
                         if url and is_number(evidence) else "")
            contact_rows += f'<div><dt>{name}</dt><dd>{value(item.get("events"))}{timestamp}</dd></div>'
        peak = mapping(contacts.get("released_apple_bowl_peak"))
        peak_value = value(peak["normal_force_n"], " N") if "normal_force_n" in peak else "—"
        if url and is_number(peak.get("sim_s")):
            peak_value += f' · <a href="{escape(url)}?t={value(peak["sim_s"])}">{value(peak["sim_s"], " s sim")}</a>'
        contact_rows += f'<div><dt>Peak released apple–bowl normal force</dt><dd>{peak_value}</dd></div>'
    contact_panel = '<div class="motion"><h3>Contact events</h3>'
    contact_panel += (f'<dl>{contact_rows}</dl><p class="detail">Sampled at 20 Hz.</p>'
                      if contact_rows else '<p class="detail">Contact telemetry unavailable for this recording.</p>')
    contact_panel += '</div>'
    link = f'<a class="link" href="{escape(url)}">Open {label.lower()} replay</a>' if url else ""
    diagnostics = mapping(trial.get("diagnostics"))
    what = (f'<p class="what">What happened: {value(diagnostics.get("summary"))}</p>'
            if diagnostics.get("summary") else "")
    note = f'<p class="detail">{escape(GENERIC_NOTE)}</p>' if trial.get("tier") == "generic" else ""
    pill = verdict or ("invalid start" if outcome == "invalid-start" else "not run")
    return (
        f'<article><div class="head"><h2>{label}</h2>'
        f'<strong class="pill {escape(pill.replace(" ", "-"))}">{escape(pill)}</strong></div>'
        f'<p class="detail">{value(trial.get("prompt", trial.get("task")))}</p>'
        f'<dl>{rows}</dl><p class="detail">{value(trial.get("reason"))}</p>{what}{note}'
        f'{quality}{contact_panel}{safety_details(trial, url)}{link}</article>'
    )


SAFETY_TEXT = {"pass": "pass", "review": "review", "block": "block", "unavailable": "unavailable"}


def safety_level(trial: object) -> str:
    safety = mapping(trial.get("safety")) if isinstance(trial, dict) else {}
    level = safety.get("level")
    return level if level in SAFETY_TEXT else "unavailable"


def safety_pill(trial: object) -> str:
    level = safety_level(trial)
    return f'<span class="safety {level}">{SAFETY_TEXT[level]}</span>'


def time_link(url: str | None, seconds: object) -> str:
    if not is_number(seconds):
        return "—"
    return f'<a href="{escape(url)}?t={value(seconds)}">{value(seconds, " s")}</a>' if url else value(seconds, " s")


def safety_details(trial: dict, url: str | None) -> str:
    """The trial's safety findings, each with its first time linked into the replay."""
    safety = mapping(trial.get("safety"))
    if not safety:
        return ""
    if safety.get("level") == "unavailable":
        return f'<div class="motion"><h3>Real-robot safety</h3><p class="detail">{value(safety.get("reason"))}</p></div>'
    rows = ""
    for item in safety.get("findings") or []:
        item = mapping(item)
        bodies = ", ".join(str(name) for name in item.get("bodies") or [])
        extra = item.get("detail") or " · ".join(part for part in (
            f'peak {item["peak_normal_force_n"]:.1f} N' if is_number(item.get("peak_normal_force_n")) else "", bodies) if part)
        rows += (f'<div><dt><span class="safety {escape(str(item.get("level")))}">{value(item.get("level"))}</span> '
                 f'{value(item.get("label"))}</dt><dd>{value(item.get("events"))} × · first {time_link(url, item.get("first_sim_s"))}'
                 f'{" · " + escape(str(extra)) if extra else ""}</dd></div>')
    speed, step = mapping(safety.get("peak_joint_speed")), mapping(safety.get("max_command_step"))
    rows += (f'<div><dt>Peak joint speed (info)</dt><dd>{value(speed.get("rad_s"), " rad/s")} · {value(speed.get("joint"))}</dd></div>'
             f'<div><dt>Largest command step (info)</dt><dd>{value(step.get("max_rad"), " rad")} · {value(step.get("joint"))}</dd></div>')
    limits = "" if safety.get("joint_limits_checked") else '<p class="detail">Joint limits unavailable: command limits were not checked.</p>'
    return f'<div class="motion"><h3>Real-robot safety</h3><dl>{rows}</dl>{limits}</div>'


def pair_report_path(path: Path, pair: dict) -> Path:
    """Where a condition's trials sit: its folder (scene, or scene/offset) next to the report."""
    folder = pair.get("folder", pair.get("scene_id"))
    if isinstance(folder, str) and folder and all(part not in ("", ".", "..") for part in folder.split("/")):
        return path.parent / folder / path.name
    return path


def condition_title(pair: dict, scenes: int) -> str:
    layout = layout_from(pair.get("object_offsets"))
    if layout is None:
        return value(pair.get("scene_id"))
    where = f'{pair.get("scene_id")} · ' if scenes > 1 else ""
    return escape(f'{where}Random start {pair.get("start", "")}: ' + "; ".join(offset.label for offset in layout))


def cell_text(trial: dict) -> str:
    return trial_verdict(trial) or ("invalid start" if trial.get("outcome") == "invalid-start" else "not run")


def verdict_cell(trial: object) -> str:
    if not isinstance(trial, dict):
        return "—"
    text = cell_text(trial)
    mode = failure_mode(trial)
    detail = f'<small>{escape(MODE_LABELS[mode])}</small>' if mode not in ("successed", "not-scored") else ""
    return (f'<span class="cell {escape(text.replace(" ", "-"))}">{escape(text)}</span>{detail}'
            f'<small>safety {safety_pill(trial)}</small>')


def arm_names(pairs: list[dict]) -> dict:
    arms = [label for label in ("baseline", "candidate") if any(isinstance(pair.get(label), dict) for pair in pairs)]
    return {label: (label.title() if len(arms) > 1 else "Trials") for label in arms}


def failure_mode_panel(pairs: list[dict]) -> str:
    """How the trials ended, by mode, with the conditions (rows of the table above) in each."""
    if len(pairs) < 2:
        return ""
    distribution = mode_distribution(pairs)
    names = arm_names(pairs)
    present = [(key, label, meaning) for key, label, meaning in MODES
               if any(key in mapping(distribution.get(arm)) for arm in names)]
    numbered = any(type(pair.get("start")) is int for pair in pairs)
    rows = ""
    for key, label, meaning in present:
        cells = ""
        for arm in names:
            entry = mapping(mapping(distribution.get(arm)).get(key))
            where = ", ".join(f"#{n}" for n in entry.get("conditions") or []) if numbered else ""
            cells += (f'<td>{value(entry.get("count", 0))}' + (f'<small>{escape(where)}</small>' if where else "") + "</td>")
        rows += f'<tr><td>{escape(label)}<small>{escape(meaning)}</small></td>{cells}</tr>'
    head = "".join(f"<th>{name}</th>" for name in names.values())
    return ('<div class="decision generalization"><h2>Failure modes</h2>'
            '<p class="detail">Read from simulator ground truth: the judged object\'s contacts and trajectory. '
            'A mode says what happened, not why the model did it.</p>'
            f'<div class="scroll"><table><thead><tr><th>Mode</th>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>')


def safety_panel(pairs: list[dict]) -> str:
    """Real-robot readiness: the worst level per arm, and every rule that fired, over all trials."""
    names = arm_names(pairs)
    if not names:
        return ""
    summary = safety_summary(pairs)
    numbered = any(type(pair.get("start")) is int for pair in pairs)
    verdicts = ""
    for arm, name in names.items():
        item = mapping(summary.get(arm))
        counts = mapping(item.get("counts"))
        tally = " · ".join(f"{value(counts[level])} {level}" for level in ("pass", "review", "block", "unavailable") if level in counts)
        level = item.get("level") if item.get("level") in SAFETY_TEXT else "unavailable"
        verdicts += f'<p><strong>{name}:</strong> <span class="safety {level}">{SAFETY_TEXT[level]}</span> <span class="muted">{tally}</span></p>'
    rows = ""
    for rule, (level, label) in SAFETY_RULES.items():
        cells, fired = "", False
        for arm in names:
            hits = []
            for index, pair in enumerate(pairs, 1):
                trial = pair.get(arm)
                findings = mapping(mapping(trial).get("safety")).get("findings") if isinstance(trial, dict) else None
                for finding in findings or []:
                    if mapping(finding).get("rule") == rule:
                        hits.append((pair.get("start") if type(pair.get("start")) is int else index, finding))
            fired = fired or bool(hits)
            peak = max((f.get("peak_normal_force_n") for _, f in hits if is_number(f.get("peak_normal_force_n"))), default=None)
            where = ", ".join(f"#{n}" for n, _ in hits) if numbered and len(pairs) > 1 else ""
            cells += (f'<td>{len(hits)} of {sum(isinstance(p.get(arm), dict) for p in pairs)} trials'
                      + (f' · peak {value(peak, " N")}' if peak is not None else "")
                      + (f'<small>{escape(where)}</small>' if where else "") + "</td>")
        rows += (f'<tr{"" if fired else " class=\"quiet\""}><td><span class="safety {level}">{level}</span> '
                 f'{escape(label)}</td>{cells}</tr>')
    head = "".join(f"<th>{name}</th>" for name in names.values())
    return ('<div class="decision generalization"><h2>Real-robot safety gate</h2>'
            f'{verdicts}<p class="detail"><b>block</b>: arms hit each other, or Dream-1 commanded a joint target outside the '
            'URDF limits or a gripper target outside 0–1 (the simulator clamps these; a real driver may not). '
            '<b>review</b>: an arm hit the table, an object or container, or itself. In this simulator the folded right '
            'arm\'s link2 and link5 touch, which may be a collision-model artifact. Forces are sampled at 20 Hz and are '
            'simulated, not calibrated. Joint speed is reported per trial without a threshold: the Dream docs give no YAM '
            'speed limit.</p>'
            f'<div class="scroll"><table><thead><tr><th>Rule</th>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>')


def generalization_panel(root: Path, path: Path, report: dict) -> str:
    """One table for the random layouts: each start's offset per object, each arm's verdict, and the
    replay links in the last column."""
    pairs = [pair for pair in report.get("scenes", []) if isinstance(pair, dict) and layout_from(pair.get("object_offsets"))]
    if not pairs:
        return ""
    info = mapping(report.get("generalization"))
    spec = mapping(info.get("random_starts"))
    judged = mapping(report.get("experiment")).get("object")
    arms = [label for label in ("baseline", "candidate") if any(label in pair for pair in pairs)]
    names = {label: (label.title() if len(arms) > 1 else "Result") for label in arms}
    objects: list[str] = []
    for pair in pairs:
        for offset in layout_from(pair["object_offsets"]):
            if offset.object not in objects:
                objects.append(offset.object)
    scenes = len({pair.get("scene_id") for pair in pairs})
    head = ("<th>#</th>" + ("<th>Scene</th>" if scenes > 1 else "")
            + "".join(f'<th>{escape(name.replace("_", " "))}{" (judged)" if name == judged else ""}<small>Δx, Δy cm</small></th>'
                      for name in objects)
            + "".join(f"<th>{names[label]}</th>" for label in arms) + "<th>Replay</th>")
    rows = ""
    for index, pair in enumerate(pairs, 1):
        moved = {offset.object: offset for offset in layout_from(pair["object_offsets"])}
        where = pair_report_path(path, pair)
        links = []
        for label in arms:
            trial = pair.get(label)
            url = replay_url(root, where, trial.get("replay"), trial.get("actions"), label) if isinstance(trial, dict) else None
            if url:
                text = "Open run replay" if len(arms) == 1 else f"Open {label} replay"
                links.append(f'<a href="{escape(url)}">{text}</a>')
        rows += (f'<tr><td>{value(pair.get("start", index))}</td>'
                 + (f'<td>{value(pair.get("scene_id"))}</td>' if scenes > 1 else "")
                 + "".join(f'<td class="num{" judged" if name == judged else ""}">'
                           f'{escape(moved[name].short) if name in moved else "—"}</td>' for name in objects)
                 + "".join(f"<td>{verdict_cell(pair.get(label))}</td>" for label in arms)
                 + f'<td>{"<br>".join(links) or "—"}</td></tr>')
    totals = mapping(info.get("totals"))
    foot = ""
    for label in arms:
        item = mapping(totals.get(label))
        counts = [(item.get("passed"), "successed"),
                  (item["scored"] - item["passed"] if is_number(item.get("scored")) and is_number(item.get("passed"))
                   else None, "failed")]
        if is_number(item.get("conditions")) and is_number(item.get("scored")) and item["conditions"] > item["scored"]:
            counts.append((item["conditions"] - item["scored"], "not run"))
        foot += "<td>" + "<br>".join(f"{value(count)} {word}" for count, word in counts) + "</td>"
    redrawn = sum(len(mapping(drawn).get("rejected") or []) for drawn in mapping(info.get("sampling")).values())
    limit = spec.get("max_offset_m")
    source = (f'{value(spec.get("count"))} random layouts, each object moved uniformly within ±{value(limit * 100)} cm, '
              f'seed {value(spec.get("seed"))}' if spec and is_number(limit) else "Random layouts")
    source += f" · {redrawn} overlapping draw{'s' if redrawn != 1 else ''} redrawn before any request" if redrawn else ""
    span = 1 + (scenes > 1) + len(objects)
    return (f'<div class="decision generalization"><h2>Generalization: random object layouts</h2>'
            f'<p class="detail">{source}. Offsets are from each object\'s default start, in MuJoCo world axes; the bowl '
            'and plate do not move. Each row is one trial, an observation rather than a success rate. '
            'The same seed reproduces the same layouts.</p>'
            f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody>'
            f'<tfoot><tr><td colspan="{span}">Total</td>{foot}<td></td></tr></tfoot></table></div></div>')


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
        first_pair = report["scenes"][0] if "scenes" in report else report
        if "candidate" in first_pair:
            label = f'{value(first_pair["baseline"].get("actions"))} vs {value(first_pair["candidate"].get("actions"))} steps'
        else:
            label = f'Single run · {value(first_pair["baseline"].get("actions"))} steps'
        if "generalization" in report:
            label += f' · {len(report["scenes"])} random layouts'
        elif "scenes" in report:
            label += f' · {len(report["scenes"])} scenes'
        if mapping(report.get("experiment")).get("comparison_variable") == "prompt":
            label = "Prompt comparison"
        links.append(f'<li><a href="/?run={quote(name)}"{active}>{label}<small>{escape(name)}</small></a></li>')
    sidebar = "".join(links) or "<li>No local comparisons</li>"
    if entry:
        path, report = entry
        pairs = report.get("scenes", [report])
        baseline = pairs[0]["baseline"]
        decision = str(report.get("decision", "Unknown"))
        single = not any("candidate" in pair for pair in pairs)
        experiment = mapping(report.get("experiment"))
        task = value(experiment.get("task_id") or baseline.get("task_id") or baseline.get("task"))
        detail = f'{task} · model {value(experiment.get("model", baseline.get("model")))} · {value(experiment.get("seconds", baseline.get("seconds")), " s")} limit'
        if "scenes" in report and single:
            detail += (f' · {value(mapping(report.get("passes")).get("baseline"))}/{value(report.get("planned_runs"))} '
                       + ("layouts" if "generalization" in report else "scenes") + " passed")
        elif "scenes" in report:
            passes = mapping(report.get("passes"))
            detail += (f' · {value(report.get("valid_pairs"))}/{value(report.get("planned_pairs"))} matched pairs'
                       f' · Baseline {value(passes.get("baseline"))} passes'
                       f' · Candidate {value(passes.get("candidate"))} passes')
        heading = DECISIONS.get(decision, decision)
        if single and "scenes" not in report:
            heading += ": " + (trial_verdict(baseline) or "not run")
        content = (
            f'<div class="decision"><h2>{escape(heading)}</h2>'
            f'<p class="detail">{value(report.get("reason"))}</p><p class="detail">{detail}</p></div>'
        ) + generalization_panel(root, path, report) + failure_mode_panel(pairs) + safety_panel(pairs)
        scene_count = len({pair.get("scene_id") for pair in pairs})
        for pair in pairs:
            scene_id = pair.get("scene_id")
            scene_path = path
            if scene_id:
                # Keep moved recordings resolvable under their condition subdirectory.
                scene_path = pair_report_path(path, pair)
                content += (f'<div class="scene"><h3>{condition_title(pair, scene_count)}</h3>'
                            f'<p class="detail">{value(pair.get("reason"))}</p>')
            if "candidate" in pair:
                content += ('<div class="cards">'
                            + trial_card(root, scene_path, "Baseline", pair["baseline"])
                            + trial_card(root, scene_path, "Candidate", pair["candidate"]) + '</div>')
            else:  # a single run is recorded in the baseline folder
                content += ('<div class="cards single">'
                            + trial_card(root, scene_path, "Run", pair["baseline"], folder="baseline") + '</div>')
            if scene_id:
                content += '</div>'
    else:
        content = ('<article><h2>Comparison unavailable</h2><p class="detail">'
                   'Select a local report, or generate a comparison with <code>compare_replays.py</code>.'
                   '</p></article>')
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regression Workbench | DreamBench</title><style>{WORKBENCH_CSS}{CSS}</style>
<header><span>◆</span>DreamBench &nbsp; <a href="/configure">Configure experiment</a></header><main>
<h1>Regression Workbench</h1>
<p class="description">Compare Dream-1 simulation runs and inspect the evidence behind each outcome.</p>
<div class="layout"><aside><nav aria-label="Experiments"><h2>Experiments</h2><ul>{sidebar}</ul></nav></aside>
<section aria-label="Comparison">{content}</section></div></main></html>"""


SIM_COMMANDS = {"/api/sim/stop": "stop", "/api/sim/reset": "reset", "/api/sim/reset_robot": "reset_robot"}
TASK_FIELDS = ("task_id", "object", "target", "scorer", "tier", "success_condition")


def recording_dir(root: Path, relative: str | None) -> Path | None:
    """A replay-* folder under the outputs root that holds a trajectory, or None."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        return None
    path = (root / relative).resolve()
    if path.is_relative_to(root) and path.name.startswith("replay-") and (path / "trajectory.jsonl").is_file():
        return path
    return None


def review_camera(query: dict) -> dict:
    """Read the 3D camera from the query string; the worker clamps it again."""
    def number(key: str) -> float | None:
        try:
            value = float(query.get(key, [""])[0])
        except ValueError:
            return None
        return value if value == value and abs(value) < 1e6 else None

    camera = {key: number(name) for key, name in (("azimuth", "az"), ("elevation", "el"), ("distance", "dist"))}
    lookat = [number(name) for name in ("lx", "ly", "lz")]
    camera = {key: value for key, value in camera.items() if value is not None}
    if all(value is not None for value in lookat):
        camera["lookat"] = lookat
    return camera


def live_replay(path: Path) -> bytes | None:
    """Rebuild replay.html from the recording, so older recordings get the current replay page."""
    track = path.parent / "trajectory.jsonl"
    if path.name not in ("replay.html", "replay-yolo.html") or not track.is_file():
        return None
    try:
        rows = [json.loads(line) for line in track.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Both pages draw the YOLO boxes on the camera views whenever the recording has them.
        detections = read_detections(path.parent)
        if path.name == "replay-yolo.html" and detections is None:
            return None
        return replay_html(rows, detections=detections,
                           detection_note=yolo_note(path.parent, detections)).encode("utf-8")
    except (OSError, ValueError):
        return None  # fall back to the saved page


def make_handler(root: Path, sim: SimWorkspace, review: ReviewRenderer | None = None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status: int, data: dict) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def local_host(self) -> str | None:
            # Reject other Host names so a DNS-rebinding page cannot read local reports.
            port = self.server.server_port
            host = self.headers.get("Host")
            return host if host in {f"127.0.0.1:{port}", f"localhost:{port}"} else None

        def read_json(self) -> object:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" or not 1 <= size <= 32768:
                raise ValueError("Expected a JSON request up to 32 KB.")
            return json.loads(self.rfile.read(size))

        def do_POST(self) -> None:
            if self.path not in ("/api/jobs", "/api/config/resolve", "/api/sim/camera", *SIM_COMMANDS):
                self.send_error(404)
                return
            host = self.local_host()
            if host is None or self.headers.get("Origin") != f"http://{host}":
                self.json_response(403, {"error": "Use the controls on the local workbench page."})
                return
            try:
                payload = self.read_json()
                if not isinstance(payload, dict):
                    raise ValueError("Send a JSON object.")
                if self.path == "/api/config/resolve":
                    config = validate_config(payload.get("config"))
                    self.json_response(200, {"task": {key: config[key] for key in TASK_FIELDS},
                                             "note": GENERIC_NOTE if config["tier"] == "generic" else None})
                    return
                if self.path == "/api/sim/camera":  # {"view": {...}} or {"reset": true}
                    view = None if payload.get("reset") is True else payload.get("view")
                    self.json_response(200, {"camera": sim.set_camera(view)})
                    return
                if self.path in SIM_COMMANDS:
                    sim.command(SIM_COMMANDS[self.path])
                    self.json_response(202, {"sent": SIM_COMMANDS[self.path]})
                    return
                if payload.get("confirm_api_credits") is not True or type(payload.get("yolo")) is not bool:
                    raise ValueError("Confirm API usage and provide the run options.")
                config = validate_config(payload.get("config"))
                if payload["yolo"] and (reason := yolo_unavailable()):
                    raise ValueError(f"{reason} Untick YOLO to run without it.")
                job_id = sim.start(config, yolo=payload["yolo"])
            except ValueError as error:
                self.json_response(400, {"error": str(error)})
                return
            except JobConflict as error:
                self.json_response(409, {"error": str(error)})
                return
            except Exception as error:  # answer in JSON rather than dropping the connection
                self.json_response(500, {"error": f"{type(error).__name__}: {error}"})
                return
            self.json_response(202, {"id": job_id})

        def do_GET(self) -> None:
            if self.local_host() is None:
                self.send_error(403, "Open the workbench at 127.0.0.1 or localhost.")
                return
            request = urlsplit(self.path)
            if request.path == "/api/health":
                self.json_response(200, {"service": "dreambench"})
                return
            if request.path == "/api/sim":
                self.json_response(200, sim.status())
                return
            if request.path == "/api/sim/viewport.jpg":
                frame = sim.viewport_frame()
                self.send_response(200 if frame else 204)
                self.send_header("Cache-Control", "no-store")
                if frame:
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                if frame:
                    self.wfile.write(frame)
                return
            if request.path in ("/api/review/summary", "/api/review/frame.jpg"):
                self.review(request.path, parse_qs(request.query))
                return
            if request.path.startswith("/api/jobs/"):
                status = sim.snapshot(request.path.removeprefix("/api/jobs/"))
                if status is None:
                    self.json_response(404, {"error": "Run status is no longer available."})
                else:
                    self.json_response(200, status)
                return
            if request.path in ("/", "/configure"):
                if request.path == "/configure":
                    payload = Path(__file__).with_name("experiment_config.html").read_bytes()
                else:
                    selected = parse_qs(request.query).get("run", [None])[0]
                    payload = dashboard(root, selected).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
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

        def review(self, endpoint: str, query: dict) -> None:
            folder = recording_dir(root, query.get("recording", [None])[0])
            if folder is None:
                self.json_response(404, {"error": "Recording not found under the workbench outputs."})
                return
            if review is None:
                self.json_response(503, {"error": "The 3D review is not enabled on this server."})
                return
            try:
                if endpoint == "/api/review/summary":
                    self.json_response(200, review.summary(folder))
                    return
                frame = int(query.get("frame", ["0"])[0])
                flags = {key: query.get(key, ["1" if key != "other" else "0"])[0] == "1"
                         for key in ("contacts", "other", "trails")}
                image = review.render(folder, frame, review_camera(query), **flags)
            except ReviewUnavailable as error:
                self.json_response(503, {"error": str(error)})
                return
            except ValueError as error:
                self.json_response(422, {"error": str(error)})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)

        def serve_artifact(self, path: Path) -> None:
            page = live_replay(path)
            if page is not None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
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
    dream_python = Path.home() / "dream-env" / "bin" / "python"
    parser.add_argument("--runner-python", type=Path,
                        default=dream_python if dream_python.is_file() else Path(sys.executable))
    parser.add_argument("--sim-viewer", action="store_true", help="Also open the desktop MuJoCo window")
    parser.add_argument("--no-sim", action="store_true",
                        help="Start the simulator only when the page asks for it, not at startup")
    args = parser.parse_args()
    review = ReviewRenderer(args.runner_python.absolute(), Path(__file__).with_name("review_worker.py").resolve())
    sim = SimWorkspace(args.outputs.resolve(), args.runner_python.absolute(),
                       Path(__file__).with_name("sim_workspace.py").resolve(), viewer=args.sim_viewer)
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sim.output_dir, sim, review)) as server:
            print(f"DreamBench · http://127.0.0.1:{server.server_port}", flush=True)
            if not args.no_sim:
                try:
                    sim.ensure_started()
                except ValueError as error:
                    print(f"Simulator not started: {error}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        sim.close()
        review.close()


if __name__ == "__main__":
    main()
