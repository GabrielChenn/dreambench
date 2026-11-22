import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch

import compare_replays
from dreamlabs.web_jobs import SimWorkspace
from web_dashboard import ThreadingHTTPServer, dashboard, make_handler


class LocalServerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        self.sim = SimWorkspace(root, Path(sys.executable), Path(__file__))
        self.addCleanup(self.sim.close)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, self.sim))
        self.addCleanup(self.server.server_close)
        Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_port

    def get(self, path: str, host: str) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        self.addCleanup(connection.close)
        connection.putrequest("GET", path, skip_host=True)
        connection.putheader("Host", host)
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        return response.status

    def test_get_accepts_loopback_hosts(self) -> None:
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            with self.subTest(host=host):
                self.assertEqual(self.get("/api/health", host), 200)

    def post(self, path: str, body: dict, origin: bool = True) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        self.addCleanup(connection.close)
        headers = {"Content-Type": "application/json", "Host": f"127.0.0.1:{self.port}"}
        if origin:
            headers["Origin"] = f"http://127.0.0.1:{self.port}"
        connection.request("POST", path, json.dumps(body), headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def config(self, baseline: str, candidate: str | None = None) -> dict:
        variable = "action_limit" if candidate is None else "prompt"
        return {"schema_version": 1, "model": "beta", "seconds": 60, "auto_stop_home_seconds": 5,
                "comparison_variable": variable,
                "baseline": {"prompt": baseline, "action_limit": 48},
                "candidate": {"prompt": candidate or baseline, "action_limit": 24 if candidate is None else 48}}

    def test_resolve_reads_task_from_prompt(self) -> None:
        for prompt, task in (("put orange into the bowl", "orange-in-bowl"),
                             ("把香蕉放到盘子上", "banana-on-plate"),
                             ("put apple into the bowl", "apple-in-bowl")):
            with self.subTest(prompt=prompt):
                status, body = self.post("/api/config/resolve", {"config": self.config(prompt)})
                self.assertEqual((status, body["task"]["task_id"]), (200, task))
                self.assertEqual(body["task"]["tier"], "calibrated")

    def test_resolve_rejects_prompts_for_different_tasks(self) -> None:
        status, body = self.post("/api/config/resolve",
                                 {"config": self.config("put apple into the bowl", "put orange into the bowl")})
        self.assertEqual(status, 400)
        self.assertIn("same task", body["error"])
        status, body = self.post("/api/config/resolve", {"config": self.config("wave hello")})
        self.assertEqual(status, 400)

    def test_sim_controls_require_page_origin(self) -> None:
        for path in ("/api/sim/stop", "/api/sim/reset", "/api/sim/reset_robot", "/api/jobs", "/api/sim/camera"):
            with self.subTest(path=path):
                self.assertEqual(self.post(path, {}, origin=False)[0], 403)

    def test_yolo_run_refused_when_yolo_missing(self) -> None:
        payload = {"config": self.config("put apple into the bowl"), "confirm_api_credits": True, "yolo": True}
        with patch("web_dashboard.yolo_unavailable", return_value="YOLO is not installed: missing weights x."):
            status, body = self.post("/api/jobs", payload)
        self.assertEqual(status, 400)
        self.assertIn("Untick YOLO", body["error"])

    def test_camera_before_the_viewport_is_ready(self) -> None:
        status, body = self.post("/api/sim/camera", {"reset": True})
        self.assertEqual(status, 400)
        self.assertIn("not ready", body["error"])

    def test_stop_without_experiment_conflicts(self) -> None:
        self.assertEqual(self.post("/api/sim/stop", {})[0], 409)

    def test_sim_status_endpoint(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", "/api/sim", headers={"Host": f"127.0.0.1:{self.port}"})
        body = json.loads(connection.getresponse().read())
        self.assertEqual((body["state"], body["job"]), ("offline", None))

    def test_get_rejects_rebound_host(self) -> None:
        for path in ("/", "/api/health", "/configure", "/artifacts/x/replay.html"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path, f"attacker.example:{self.port}"), 403)


class DashboardRobustnessTests(unittest.TestCase):
    def test_malformed_report_sections_render(self) -> None:
        trial = {"outcome": "success", "actions": 24, "contact_summary": {
            "sampled_frames": 3, "events": {"arm_table": "bad"}, "released_apple_bowl_peak": []}}
        report = {"decision": "keep-baseline", "experiment": ["bad"], "passes": None,
                  "scenes": [{"scene_id": "yam-tabletop-default", "baseline": trial, "candidate": trial}]}
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "comparison-abc123"
            folder.mkdir()
            (folder / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
            page = dashboard(Path(directory), None)
        self.assertIn("Keep baseline", page)
        self.assertIn("Arm–table collisions", page)


class SingleRunDisplayTests(unittest.TestCase):
    def test_single_run_report(self) -> None:
        trial = {"outcome": "success", "verdict": "successed", "actions": 48, "reason": "Placed.",
                 "prompt": "put orange into the bowl"}
        report = {"decision": "single-run", "reason": "Placed.", "baseline": trial,
                  "experiment": {"comparison_variable": None, "model": "beta", "seconds": 60}}
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "comparison-abc123"
            folder.mkdir()
            (folder / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
            page = dashboard(Path(directory), None)
        self.assertIn("Single run: successed", page)
        self.assertIn("Single run · 48 steps", page)
        self.assertIn("<h2>Run</h2>", page)
        self.assertNotIn("Candidate", page)


class VerdictDisplayTests(unittest.TestCase):
    def render(self, baseline: dict, candidate: dict) -> str:
        report = {"decision": "keep-baseline", "reason": "r", "baseline": baseline, "candidate": candidate}
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "comparison-abc123"
            folder.mkdir()
            (folder / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
            return dashboard(Path(directory), None)

    def test_verdicts_and_diagnostics(self) -> None:
        page = self.render(
            {"outcome": "success", "verdict": "successed", "stop_reason": "home",
             "diagnostics": {"summary": "The orange ended resting in the bowl with the arms and table clear."}},
            {"outcome": "task-failure", "verdict": "timeout stopped", "stop_reason": "home", "tier": "generic"})
        self.assertIn('class="pill successed">successed', page)
        self.assertIn('class="pill failed">failed', page)  # an older report's "timeout stopped" reads as failed
        self.assertNotIn("timeout stopped", page)
        self.assertIn("What happened: The orange ended resting in the bowl", page)
        self.assertIn("Arms back home", page)
        self.assertIn("Generic placement rule", page)

    def test_older_reports_get_a_verdict_from_their_outcome(self) -> None:
        page = self.render({"outcome": "success"}, {"outcome": "not-run"})
        self.assertIn('class="pill successed">successed', page)
        self.assertIn('class="pill not-run">not run', page)


class CompareRunValidationTests(unittest.TestCase):
    def run_cli(self, *argv: str) -> None:
        with TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["compare_replays.py", "--run", "--output-dir", directory, *argv]), \
                patch("compare_replays.run_pair", side_effect=AssertionError("must not start a run")), \
                contextlib.redirect_stderr(io.StringIO()):
            compare_replays.main()

    def test_run_rejects_values_the_config_path_rejects(self) -> None:
        for argv in (["--seconds", "1000"], ["--seconds", "nan"], ["--model", "beta 2"],
                     ["--baseline-actions", "24"], ["--scene", "yam-tabletop-default"] * 2,
                     ["--prompt", "wave hello"], ["--home-seconds", "0.2"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                self.run_cli(*argv)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
