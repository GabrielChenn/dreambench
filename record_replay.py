"""Record and replay a Dream YAM Sim run on a shared simulation timeline."""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import av
import numpy as np


CAMERAS = ("top", "left", "right")


def write_replay(path: Path, rows: list[dict], *, detections: list[dict] | None = None) -> None:
    # Keep replay self-contained apart from its sibling video; file:// needs no server.
    payload = json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
    detection_payload = json.dumps(detections, allow_nan=False).replace("<", "\\u003c")
    path.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<title>DreamBench Replay</title>
<style>
body{font:16px system-ui;margin:2rem auto;max-width:1200px;padding:0 1rem}
.video-wrap{position:relative}
video{display:block;width:100%;background:#111}
canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none} pre{background:#f4f4f4;padding:1rem;overflow:auto}
.cameras{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;margin-bottom:.4rem}
input{width:7em} output{margin-left:1em}
</style>
<h1>DreamBench Replay</h1>
<p>Simulation time starts at zero. Inference waits are visible in the log's wall time.</p>
<div class="cameras"><span>Overhead</span><span>Left wrist</span><span>Right wrist</span></div>
<div class="video-wrap"><video id="video" src="video.mp4" controls preload="metadata"></video><canvas id="boxes" hidden></canvas></div>
<p><label>Simulation time (s) <input id="seek" type="number" min="0" step="0.05"></label>
<button id="jump">Jump</button><output id="position"></output></p>
<pre id="log"></pre>
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
const detections = JSON.parse(document.getElementById('detections').textContent);
const canvas = document.getElementById('boxes');
const context = canvas.getContext('2d');
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
  position.textContent = row ? row.sim_seconds.toFixed(2) + ' s · frame ' + index : 'No frames';
  const detection = detections?.[index];
  if (detections) {
    const counts = detection?.ran
      ? ['top', 'left', 'right'].map(camera => detection.cameras[camera].length).join(' / ')
      : 'not run';
    position.textContent += ' · YOLO detections (overhead / left / right): ' + counts;
  }
  draw(detection);
  log.textContent = row ? JSON.stringify(detection ? {...row, detection} : row, null, 2) : '';
}
document.getElementById('jump').onclick = () => {
  const seconds = seek.valueAsNumber;
  if (!rows.length || !Number.isFinite(seconds)) return;
  video.pause();
  video.currentTime = Math.max(0, Math.min(rows.at(-1).sim_seconds, seconds));
  show(video.currentTime);
};
video.addEventListener('loadedmetadata', () => show(video.currentTime));
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
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
    from dream_control import main as run_dream
    from yam_env import YamEnv

    output = args.output_dir / f"replay-{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Recording to {output.resolve()}", flush=True)
    rows = []
    stream = None
    origin_wall = None
    origin_sim = None

    class RecordingEnv(YamEnv):
        targets = None

        def step(self, left, right):
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
                "apple_contacts": self.apple_contacts(),
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
        metadata.update(
            status=status, frames=len(rows), fps=20, includes_warmup=True,
            video_camera_order=CAMERAS, video_layout="horizontal",
        )
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if rows:
            write_replay(output / "replay.html", rows)
            print(f"Replay: {(output / 'replay.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
