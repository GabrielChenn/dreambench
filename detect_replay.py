"""Run YOLO object detection in an existing three-camera Dream YAM Sim recording."""

import argparse
import json
import math
import multiprocessing
from multiprocessing.connection import Connection
import os
import sys
from pathlib import Path

import numpy as np



LOCAL_DEPS = Path(__file__).resolve().parent / ".venv" / "yolo-deps"
LOCAL_WEIGHTS = LOCAL_DEPS.parent / "yolo11n.pt"
CLASSES = {"apple", "orange", "banana", "bowl", "bottle"}


def detect(model, tiles: list[np.ndarray], cameras: tuple[str, ...], confidence: float, device: str) -> dict:
    # Ultralytics accepts BGR for NumPy inputs. Each box is local to its source camera.
    images = [np.ascontiguousarray(tile[:, :, ::-1]) for tile in tiles]
    results = model.predict(images, conf=confidence, device=device, verbose=False)
    return {
        camera: [
            {
                "label": result.names[int(class_id)],
                "bbox_xyxy_px": coordinates,
                "confidence": score,
            }
            for coordinates, score, class_id in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
            )
            if result.names[int(class_id)] in CLASSES
        ]
        for camera, result in zip(cameras, results, strict=True)
    }


def inference_worker(connection: Connection, weights: Path, confidence: float, device: str) -> None:
    """Load OpenCV through Ultralytics in a process that never imports PyAV."""
    with connection:
        try:
            if LOCAL_DEPS.is_dir():
                sys.path.insert(0, str(LOCAL_DEPS))
                os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_DEPS.parent / "yolo-config"))
            from ultralytics import YOLO, __version__

            model = YOLO(str(weights), task="detect")
            connection.send((True, {
                "name": "Ultralytics YOLO", "version": __version__, "weights": weights.name,
                "confidence_threshold": confidence,
            }))
            while (request := connection.recv()) is not None:
                tiles, cameras = request
                connection.send((True, detect(model, tiles, cameras, confidence, device)))
        except EOFError:
            return
        except Exception as error:
            connection.send((False, f"{type(error).__name__}: {error}"))


def receive(connection: Connection) -> dict:
    try:
        ok, value = connection.recv()
    except (EOFError, OSError) as error:
        raise RuntimeError("YOLO worker exited before sending a result.") from error
    if not ok:
        raise RuntimeError(value)
    return value


def annotate(folder: Path, every: int, connection: Connection, detector: dict) -> None:
    import av
    from record_replay import CAMERAS, write_replay

    metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if metadata.get("video_camera_order") != list(CAMERAS) or metadata.get("video_layout") != "horizontal":
        raise ValueError("This run does not declare a supported three-camera video layout.")
    rows = [
        json.loads(line)
        for line in (folder / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if not rows:
        raise ValueError("The recording has no trajectory rows.")

    records = []
    with av.open(str(folder / "video.mp4")) as container:
        frames = iter(container.decode(video=0))
        for index, row in enumerate(rows):
            frame = next(frames, None)
            if frame is None or row["frame"] != index or frame.time is None:
                raise ValueError(f"Video and trajectory are misaligned at frame {index}.")
            if abs(frame.time - row["sim_seconds"]) > 0.001 or frame.width % len(CAMERAS):
                raise ValueError(f"Unexpected video timestamp or dimensions at frame {index}.")
            ran = index % every == 0
            cameras = {}
            if ran:
                tiles = np.split(frame.to_ndarray(format="rgb24"), len(CAMERAS), axis=1)
                connection.send((tiles, CAMERAS))
                cameras = receive(connection)
            records.append({
                "frame": index, "sim_seconds": row["sim_seconds"],
                "detector": detector, "ran": ran,
                "camera_size_px": [frame.width // len(CAMERAS), frame.height],
                "cameras": cameras,
            })
        if next(frames, None) is not None:
            raise ValueError("Video contains additional frames without trajectory rows.")

    path = folder / "detections-yolo.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    replay = folder / "replay-yolo.html"
    write_replay(replay, rows, detections=records)
    print(f"Detections: {path.resolve()}\nReplay: {replay.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="outputs/replay-* directory")
    parser.add_argument("--every", type=int, default=1, help="Run detection once per N frames")
    parser.add_argument("--weights", type=Path, default=LOCAL_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or mps")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be at least 1.")
    if not math.isfinite(args.confidence) or not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1.")
    if not args.weights.is_file():
        parser.error(f"YOLO weights missing: {args.weights}. Pass a local .pt file with --weights.")
    # Spawn a clean interpreter so the detector cannot inherit PyAV's native libraries.
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(
        target=inference_worker,
        args=(child, args.weights.resolve(), args.confidence, args.device),
    )
    try:
        with parent, child:
            worker.start()
            child.close()
            detector = receive(parent)
            annotate(args.recording, args.every, parent, detector)
            parent.send(None)
        worker.join(timeout=5)
        if worker.exitcode != 0:
            raise RuntimeError("YOLO worker failed to shut down cleanly.")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Detection failed: {error}\n")
    finally:
        if worker.pid is not None:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
            worker.join()


if __name__ == "__main__":
    main()
