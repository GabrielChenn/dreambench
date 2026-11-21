"""Record and replay a Dream YAM Sim run on a shared simulation timeline."""

import argparse
from contextlib import ExitStack
from html import escape
import importlib.util
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import av
import numpy as np

from dreamlabs.contact_events import sample_contacts
from dreamlabs.scenarios import DEFAULT_SCENE_ID, SCENES
from dreamlabs.web_theme import WORKBENCH_CSS


CAMERAS = ("top", "left", "right")


CAMERA_LABELS = {"top": "Overhead", "left": "Left wrist", "right": "Right wrist"}
REPLAY_CSS = """
.panel{background:var(--surface);border:1px solid var(--line);border-radius:.9rem;padding:1.2rem;margin-top:1.25rem}
header nav{display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap}
header a{font-weight:650}
.video-wrap{position:relative;background:#080e17;border-radius:.55rem;overflow:hidden}
video{display:block;width:100%;height:auto}
canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.cameras{display:flex;margin-bottom:.6rem;gap:.4rem}
.cameras span{flex:1;text-align:center;color:var(--muted);font-size:.875rem}
.controls{display:flex;align-items:end;flex-wrap:wrap;gap:.65rem;margin-top:1rem}
.controls label{display:grid;gap:.35rem;color:var(--muted);font-size:.875rem}
input,button{font:inherit;border-radius:.45rem;padding:.45rem .7rem}
input{width:8rem;color:var(--ink);background:#0c131e;border:1px solid #40516b;font-variant-numeric:tabular-nums}
button{color:#081d19;background:var(--accent);border:1px solid transparent;font-weight:700;cursor:pointer}
button:disabled{opacity:.55;cursor:default}
output{display:block;color:var(--muted);margin-top:.75rem;font-size:.9rem;font-variant-numeric:tabular-nums}
.frame-tools{display:flex;gap:.65rem;align-items:center;flex-wrap:wrap;margin:.7rem 0}
.secondary{background:#26364a;color:var(--ink)}
.table-wrap{overflow:auto;margin:.75rem 0}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:.4rem .55rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child{text-align:left}
thead th{color:var(--muted);font-weight:650;font-size:.875rem}
pre{background:#0c131e;border:1px solid var(--line);color:#d3e5f6;border-radius:.55rem;padding:1rem;overflow:auto;margin:0;max-height:28rem}
@media(max-width:540px){.panel{padding:.8rem}.cameras span{font-size:.78rem}}
"""


def write_replay(
    path: Path,
    rows: list[dict],
    *,
    detections: list[dict] | None = None,
    cameras: tuple[str, ...] = CAMERAS,
) -> None:
    # Keep replay self-contained apart from its sibling video; file:// needs no server.
    payload = json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
    detection_payload = json.dumps(detections, allow_nan=False).replace("<", "\\u003c")
    camera_labels = "".join(f"<span>{escape(CAMERA_LABELS.get(camera, camera))}</span>" for camera in cameras)
    path.write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DreamBench Replay</title><style>{WORKBENCH_CSS}{REPLAY_CSS}</style>
<header><nav aria-label="Workbench"><strong><span>◆</span>DreamBench</strong><a id="back" hidden>Back to results</a></nav></header>
<main><h1>Simulation replay</h1>
<p class="description">Inspect the camera views and telemetry on the same simulation timeline.</p>
<section class="panel" aria-labelledby="video-title">
<h2 id="video-title">Camera views</h2>
<div class="cameras">{camera_labels}</div>
<div class="video-wrap"><video id="video" src="video.mp4" controls preload="metadata"></video><canvas id="boxes" hidden></canvas></div>
<form id="seek-form" class="controls"><label>Simulation time (s)<input id="seek" type="number" min="0" step="any" required></label>
<button id="jump" type="submit" disabled>Jump</button>
<button id="previous-frame" class="secondary" type="button" disabled>Previous frame</button>
<button id="next-frame" class="secondary" type="button" disabled>Next frame</button></form>
<output id="position"></output></section>
<section class="panel" aria-labelledby="action-title">
<h2 id="action-title">Frame action</h2>
<p id="action-context" class="description"></p>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Joint</th><th scope="col">Left target</th><th scope="col">Left measured</th>
<th scope="col">Right target</th><th scope="col">Right measured</th></tr></thead>
<tbody id="action-values"></tbody></table></div>
<div class="frame-tools"><button id="export-frame" type="button" class="secondary" disabled>Download frame JSON</button></div>
<p id="reproduction" class="description"></p>
</section>
<section class="panel" aria-labelledby="log-title">
<h2 id="log-title">Frame telemetry</h2>
<pre id="log"></pre>
</section></main>
<script id="rows" type="application/json">"""
        + payload
        + """</script>
<script id="detections" type="application/json">"""
        + detection_payload
        + """</script>
<script>
const rows = JSON.parse(document.getElementById('rows').textContent);
const video = document.getElementById('video');
const seek = document.getElementById('seek');
const position = document.getElementById('position');
const log = document.getElementById('log');
const actionValues = document.getElementById('action-values');
const actionContext = document.getElementById('action-context');
const reproduction = document.getElementById('reproduction');
const previous = document.getElementById('previous-frame');
const next = document.getElementById('next-frame');
const exportFrame = document.getElementById('export-frame');
let currentIndex = -1;
let mediaReady = false;
const jointNames = ['Joint 1 (°)', 'Joint 2 (°)', 'Joint 3 (°)', 'Joint 4 (°)',
                    'Joint 5 (°)', 'Joint 6 (°)', 'Gripper opening (0–1)'];
function showAction(row) {
  actionValues.replaceChildren();
  actionContext.textContent = row
    ? 'Targets are sent before the step. Measured values are sampled after the step.'
    : 'No telemetry at this time.';
  reproduction.textContent = !row ? '' : row.physics_before_step
    ? 'Physics snapshot available. Download the frame and run python record_replay.py --reproduce-frame <downloaded.json>.'
    : 'This recording contains joint telemetry. Physics snapshots are available in new recordings.';
  exportFrame.disabled = !row;
  if (!row) return;
  const target = row.commanded_joint_rad_and_gripper;
  const measured = row.measured_joint_rad_and_gripper;
  const radiansToDegrees = 180 / Math.PI;
  jointNames.forEach((name, joint) => {
    const tr = actionValues.insertRow();
    const label = document.createElement('th');
    label.scope = 'row';
    label.textContent = name;
    tr.append(label);
    for (const vector of [target?.left, measured?.left, target?.right, measured?.right]) {
      const value = vector?.[joint];
      const displayValue = joint < jointNames.length - 1 ? value * radiansToDegrees : value;
      tr.insertCell().textContent = Number.isFinite(displayValue) ? displayValue.toFixed(4) : '—';
    }
  });
}

const detections = JSON.parse(document.getElementById('detections').textContent);
const canvas = document.getElementById('boxes');
const context = canvas.getContext('2d');
const back = document.getElementById('back');
const [, route, experiment] = location.pathname.split('/');
if (route === 'artifacts' && experiment?.startsWith('comparison-')) {
  back.href = '/?' + new URLSearchParams({run: experiment});
  back.hidden = false;
} else if (location.pathname.startsWith('/artifacts/')) {
  back.href = '/';
  back.hidden = false;
}
canvas.hidden = !detections;
function draw(record) {
  if (!detections || !video.videoWidth) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!record?.ran) return;
  const width = record.camera_size_px[0];
  context.font = 'bold 18px system-ui';
  context.lineWidth = 3;
  for (const [cameraIndex, camera] of ['top', 'left', 'right'].entries()) {
    for (const object of record.cameras[camera]) {
      const [x1, y1, x2, y2] = object.bbox_xyxy_px;
      const x = cameraIndex * width + x1;
      const color = {apple: '#ff6363', orange: '#ffab45', banana: '#fff16b', bowl: '#67dbff'}[object.label] ?? '#a7ff83';
      const label = object.confidence == null ? object.label :
        object.label + ' ' + Math.round(object.confidence * 100) + '%';
      context.strokeStyle = color;
      context.strokeRect(x, y1, x2 - x1, y2 - y1);
      const labelY = Math.max(0, y1 - 23);
      context.fillStyle = '#141414';
      context.fillRect(x, labelY, context.measureText(label).width + 8, 22);
      context.fillStyle = color;
      context.fillText(label, x + 4, labelY + 17);
    }
  }
}
function show(seconds) {
  const index = Math.min(rows.length - 1, Math.max(0, Math.floor(seconds * 20 + 1e-6)));
  const row = rows[index];
  previous.disabled = !mediaReady || index <= 0;
  next.disabled = !mediaReady || index < 0 || index >= rows.length - 1;
  if (index === currentIndex) { draw(detections?.[index]); return; }
  currentIndex = index;
  showAction(row);
  position.textContent = row ? row.sim_seconds.toFixed(2) + ' s · frame ' + index : 'No frames';
  const detection = detections?.[index];
  if (Array.isArray(row?.contact_samples)) {
    const labels = {grasp_contact: 'Grasp contact', arm_table: 'Arm–table', arm_arm: 'Arm–arm', apple_bowl: 'Apple–bowl'};
    const contacts = [...new Set(row.contact_samples.map(contact => labels[contact.category]))];
    position.textContent += ' · ' + (contacts.join(', ') || 'No tracked contacts');
  }
  if (detections) {
    const counts = detection?.ran
      ? ['top', 'left', 'right'].map(camera => detection.cameras[camera].length).join(' / ')
      : 'not run';
    position.textContent += ' · YOLO detections (overhead / left / right): ' + counts;
  }
  draw(detection);
  if (row) {
    const {physics_before_step, ...telemetry} = row;
    if (physics_before_step) telemetry.physics_snapshot_available = true;
    if (detection) telemetry.detection = detection;
    log.textContent = JSON.stringify(telemetry, null, 2);
  } else {
    log.textContent = '';
  }
}
function jumpTo(seconds) {
  if (!mediaReady || !rows.length || !Number.isFinite(seconds)) return;
  video.pause();
  video.currentTime = Math.max(0, Math.min(rows.at(-1).sim_seconds, seconds));
  seek.value = video.currentTime.toFixed(2);
  show(video.currentTime);
}
document.getElementById('seek-form').addEventListener('submit', event => {
  event.preventDefault();
  jumpTo(seek.valueAsNumber);
});
previous.onclick = () => jumpTo(rows[currentIndex - 1]?.sim_seconds);
next.onclick = () => jumpTo(rows[currentIndex + 1]?.sim_seconds);
exportFrame.onclick = () => {
  const row = rows[currentIndex];
  if (!row) return;
  const detection = detections?.[currentIndex];
  const data = JSON.stringify(detection ? {...row, detection} : row, null, 2);
  const url = URL.createObjectURL(new Blob([data], {type:'application/json'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'dreambench-frame-' + row.frame + '.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
video.addEventListener('loadedmetadata', () => {
  mediaReady = true;
  document.getElementById('jump').disabled = !rows.length;
  const requested = new URLSearchParams(location.search).get('t');
  const seconds = Number(requested);
  if (rows.length && requested !== null && Number.isFinite(seconds)) {
    video.currentTime = Math.max(0, Math.min(rows.at(-1).sim_seconds, seconds));
    seek.value = video.currentTime.toFixed(2);
  }
  show(video.currentTime);
});
video.addEventListener('timeupdate', () => show(video.currentTime));
video.addEventListener('seeked', () => show(video.currentTime));
if (video.requestVideoFrameCallback) {
  const frame = (_, metadata) => {
    show(metadata.mediaTime);
    video.requestVideoFrameCallback(frame);
  };
  video.requestVideoFrameCallback(frame);
}
show(0);
</script></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewport-jpeg", type=Path, help="Publish the latest MuJoCo viewport JPEG")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scene", choices=tuple(SCENES), default=DEFAULT_SCENE_ID)
    parser.add_argument("--reproduce-frame", type=Path, help="Re-run one exported simulation step offline")
    parser.add_argument("--check-reset", action="store_true", help="Check this scene twice without rendering or Dream-1")
    parser.add_argument("--task", default="put apple into the bowl")
    parser.add_argument("--model", default="beta")
    parser.add_argument("--actions", type=int, choices=range(1, 49), default=48, metavar="1–48")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--headless", action="store_true", help="Record without opening the MuJoCo viewer")
    parser.add_argument("--dry-run", action="store_true", help="Record simulator warm-up without calling Dream-1")
    args = parser.parse_args()

    spec = importlib.util.find_spec("dream_examples.yam_mujoco")
    if spec is None or not spec.submodule_search_locations:
        parser.error("Dream YAM Sim is missing. Run with ~/dream-env/bin/mjpython.")
    # The upstream example imports sibling modules by their bare module names.
    sys.path.insert(0, str(next(iter(spec.submodule_search_locations))))
    import mujoco

    from dream_control import main as run_dream
    from yam_env import VIEWER_AZIMUTH, VIEWER_DISTANCE, VIEWER_ELEVATION, VIEWER_LOOKAT, YamEnv

    from dreamlabs.frame_restore import STATE_SCHEMA, integration_state, model_digest, reproduce_step

    if args.reproduce_frame:
        env = YamEnv()
        try:
            row = json.loads(args.reproduce_frame.read_text(encoding="utf-8"))
            reproduced = reproduce_step(env, row)
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(str(error))
        finally:
            env.close()
        print(json.dumps(reproduced, indent=2))
        if not reproduced["matches_recording"]:
            raise SystemExit(1)
        return

    scene = SCENES[args.scene]

    class SceneEnv(YamEnv):
        def reset(self):
            state = super().reset()
            if any(scene.apple_offset_xyz_m):
                self.data.joint("apple_free").qpos[:3] += scene.apple_offset_xyz_m
                mujoco.mj_forward(self.model, self.data)
            return state

    if args.check_reset:
        env = SceneEnv()
        try:
            def snapshot():
                return {"qpos": env.data.qpos.tolist(), "qvel": env.data.qvel.tolist(),
                        "apple_xyz_m": env.data.body("apple").xpos.tolist(),
                        "bowl_xyz_m": env.data.body("bowl").xpos.tolist()}

            env.reset()
            first = snapshot()
            env.data.joint("apple_free").qpos[0] += 0.01
            state = env.state()
            env.step(state["left"], state["right"])
            env.reset()
            if snapshot() != first:
                raise RuntimeError(f"Scene reset is not repeatable: {args.scene}")
            print(json.dumps({
                "scene_id": args.scene, "apple_offset_xyz_m": scene.apple_offset_xyz_m,
                "checks": 2, "identical_initial_state": True,
                "apple_xyz_m": first["apple_xyz_m"], "bowl_xyz_m": first["bowl_xyz_m"],
            }, indent=2))
        finally:
            env.close()
        return

    output = args.output_dir / f"replay-{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Recording to {output.resolve()}", flush=True)
    rows = []
    stream = None
    origin_wall = None
    origin_sim = None

    viewport = None
    if args.viewport_jpeg is not None:
        from PIL import Image

        viewport = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(viewport)
        viewport.lookat[:] = VIEWER_LOOKAT
        viewport.distance = VIEWER_DISTANCE
        viewport.azimuth = VIEWER_AZIMUTH
        viewport.elevation = VIEWER_ELEVATION

    class RecordingEnv(SceneEnv):
        targets = None
        before_step = None
        model_sha256 = None
        last_viewport = float("-inf")

        def step(self, left, right):
            if self.model_sha256 is None:
                self.model_sha256 = model_digest(self.model)
            self.before_step = {
                "schema": STATE_SCHEMA, "mujoco_version": mujoco.__version__,
                "model_sha256": self.model_sha256, "values": integration_state(self.model, self.data).tolist(),
            }
            state = super().step(left, right)
            self.targets = {"left": left.tolist(), "right": right.tolist()}
            return state

        def apple_contacts(self) -> list[str]:
            apple = self.model.body("apple").id
            touching = set()
            for contact in self.data.contact[: self.data.ncon]:
                if contact.dist > 0:
                    continue
                first, second = (int(self.model.geom_bodyid[g]) for g in contact.geom)
                if first == apple:
                    touching.add(self.model.body(second).name)
                elif second == apple:
                    touching.add(self.model.body(first).name)
            return sorted(touching)

        def images(self):
            nonlocal stream, origin_wall, origin_sim
            frames = super().images()
            now = time.monotonic()
            if viewport is not None and now - self.last_viewport >= 0.16:
                self.renderer.update_scene(self.data, camera=viewport)
                pending = args.viewport_jpeg.with_suffix(".tmp")
                Image.fromarray(self.renderer.render()).save(pending, format="JPEG", quality=80)
                pending.replace(args.viewport_jpeg)
                self.last_viewport = now
            sim = float(self.data.time)
            if origin_wall is None:
                origin_wall, origin_sim = now, sim
            elif rows and abs((sim - origin_sim) - rows[-1]["sim_seconds"]) <= 1e-6:
                # The live viewer refreshes while inference holds simulation time.
                return frames
            elif abs((sim - origin_sim) - len(rows) / 20) > 1e-6:
                raise RuntimeError("Expected one camera sample per 50 ms simulator step.")
            # Pack one sample from each camera into the same video frame.
            mosaic = np.concatenate([frames[camera] for camera in CAMERAS], axis=1)
            frame = av.VideoFrame.from_ndarray(mosaic, format="rgb24")
            if stream is None:
                stream = video.add_stream("libx264", rate=20)
                stream.width, stream.height = frame.width, frame.height
                stream.pix_fmt = "yuv420p"
                stream.options = {"preset": "ultrafast", "crf": "23"}
            row = {
                "frame": len(rows),
                "sim_seconds": len(rows) / 20,
                "wall_seconds": round(now - origin_wall, 6),
                "measured_joint_rad_and_gripper": {
                    side: values.tolist() for side, values in self.state().items()
                },
                "commanded_joint_rad_and_gripper": self.targets,
                "physics_before_step": self.before_step,
                "apple_contacts": self.apple_contacts(),
                "contact_samples": sample_contacts(self.model, self.data),
                "object_xyz_m": {
                    name: self.data.body(name).xpos.tolist()
                    for name in ("apple", "orange", "bowl")
                },
            }
            line = json.dumps(row, allow_nan=False)
            for packet in stream.encode(frame):
                video.mux(packet)
            log.write(line + "\n")
            log.flush()
            rows.append(row)
            return frames

    status = "interrupted"
    try:
        with ExitStack() as stack:
            log = stack.enter_context((output / "trajectory.jsonl").open("w", encoding="utf-8"))
            video = stack.enter_context(av.open(str(output / "video.mp4"), mode="w"))
            env = RecordingEnv()
            stack.callback(env.close)
            options = [
                f"--task={args.task}", f"--model={args.model}",
                "--actions", str(args.actions), "--seconds", str(args.seconds),
            ]
            if args.headless or args.dry_run:
                options.append("--headless")
            if args.dry_run:
                options.append("--dry-run")
            try:
                run_dream(options, shared_env=env)
                status = "finished"
            finally:
                if stream is not None:
                    for packet in stream.encode():
                        video.mux(packet)
    finally:
        metadata = vars(args).copy()
        metadata.pop("output_dir")
        metadata.pop("viewport_jpeg")
        metadata.update(
            status=status, frames=len(rows), fps=20, includes_warmup=True,
            video_camera_order=CAMERAS, video_layout="horizontal",
            scene_id=args.scene, apple_offset_xyz_m=scene.apple_offset_xyz_m,
        )
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if rows:
            write_replay(output / "replay.html", rows)
            print(f"Replay: {(output / 'replay.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
