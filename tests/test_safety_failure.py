"""Failure modes and the real-robot safety gate, from synthetic telemetry. No API calls."""

import importlib.util
import unittest

from dreamlabs.failure_modes import failure_mode, mode_distribution
from dreamlabs.safety import safety_report, safety_summary
import web_dashboard

HAS_SIM = importlib.util.find_spec("dream_examples") is not None and importlib.util.find_spec("mujoco") is not None
HOME = [0.0] * 6 + [1.0]
LIMITS = {side: [(-2.6, 3.1), (0.0, 3.65), (0.0, 3.1), (-1.57, 1.57), (-1.57, 1.57), (-2.09, 2.09)]
          for side in ("left", "right")}


def row(t: float, contacts=(), left=None, right=None, target_left=None, target_right=None) -> dict:
    return {"sim_seconds": t,
            "measured_joint_rad_and_gripper": {"left": left or HOME, "right": right or HOME},
            "commanded_joint_rad_and_gripper": {"left": target_left or left or HOME, "right": target_right or right or HOME},
            "contact_samples": [{"category": category, "bodies": list(bodies), "normal_force_n": force}
                                for category, bodies, force in contacts]}


def trial(outcome="task-failure", verdict="failed", stage=None, final_contacts=(), safety=None) -> dict:
    return {"outcome": outcome, "verdict": verdict, "safety": safety,
            "diagnostics": {"stage": stage, "final_contacts": list(final_contacts)} if stage else None}


class FailureModeTests(unittest.TestCase):
    def test_modes_from_diagnosis(self) -> None:
        cases = [(trial("success", "successed", "on-target"), "successed"),
                 (trial(stage="untouched"), "never-reached"),
                 (trial(stage="moved-without-arm-contact"), "knocked-away"),
                 (trial(stage="touched-not-lifted"), "grasp-failed"),
                 (trial(stage="lifted-not-placed", final_contacts=["right_tip_left"]), "still-holding"),
                 (trial(stage="lifted-not-placed", final_contacts=["world"]), "dropped-or-misplaced"),
                 (trial(stage="touching-target"), "not-released"),
                 (trial(stage="on-target"), "unsettled"),
                 (trial(), "unknown"),
                 (trial("invalid-start", None), "not-scored"),
                 (trial("run-error"), "not-scored")]
        for item, mode in cases:
            with self.subTest(mode=mode):
                self.assertEqual(failure_mode(item), mode)

    def test_distribution_counts_and_names_conditions(self) -> None:
        pairs = [{"start": 1, "baseline": trial(stage="untouched")},
                 {"start": 2, "baseline": trial("success", "successed")},
                 {"start": 3, "baseline": trial(stage="untouched")}]
        self.assertEqual(mode_distribution(pairs), {"baseline": {
            "successed": {"count": 1, "conditions": [2]}, "never-reached": {"count": 2, "conditions": [1, 3]}}})


class SafetyTests(unittest.TestCase):
    def test_no_contact_telemetry_is_unavailable(self) -> None:
        rows = [row(0.0)]
        del rows[0]["contact_samples"]
        self.assertEqual(safety_report(rows, LIMITS)["level"], "unavailable")
        self.assertEqual(safety_report([], LIMITS)["level"], "unavailable")

    def test_clean_run_passes_and_reports_speed(self) -> None:
        moved = [0.1] + HOME[1:]
        report = safety_report([row(0.0), row(0.05, left=moved)], LIMITS)
        self.assertEqual((report["level"], report["findings"]), ("pass", []))
        self.assertAlmostEqual(report["peak_joint_speed"]["rad_s"], 2.0)
        self.assertEqual(report["peak_joint_speed"]["joint"], "left_joint1")
        self.assertAlmostEqual(report["max_command_step"]["max_rad"], 0.1)

    def test_collisions_count_events_and_set_the_level(self) -> None:
        table = ("arm_table", ("left_link3", "world"), 4.0)
        rows = [row(0.0, [table]), row(0.05, [table, table]), row(0.10), row(0.15, [table])]
        report = safety_report(rows, LIMITS)
        self.assertEqual(report["level"], "review")
        finding = report["findings"][0]
        self.assertEqual((finding["rule"], finding["events"], finding["first_sim_s"], finding["peak_normal_force_n"]),
                         ("arm_table", 2, 0.0, 8.0))
        arms = safety_report(rows + [row(0.2, [("arm_arm", ("left_link5", "right_link5"), 12.0)])], LIMITS)
        self.assertEqual(arms["level"], "block")
        self.assertEqual(arms["findings"][0]["rule"], "arm_arm")  # blocking findings first

    def test_commands_outside_limits_block(self) -> None:
        beyond = [0.0, -0.2] + HOME[2:]
        report = safety_report([row(0.0), row(0.05, target_right=beyond)], LIMITS)
        self.assertEqual(report["level"], "block")
        self.assertEqual(report["findings"][0]["rule"], "command_limit")
        self.assertIn("right_joint2 by up to 0.2 rad", report["findings"][0]["detail"])
        gripper = safety_report([row(0.0, target_left=HOME[:6] + [1.2])], LIMITS)
        self.assertEqual((gripper["level"], gripper["findings"][0]["rule"]), ("block", "gripper_range"))

    def test_without_limits_joint_commands_are_not_judged(self) -> None:
        report = safety_report([row(0.0, target_right=[0.0, -0.2] + HOME[2:])], None)
        self.assertEqual(report["level"], "pass")
        self.assertFalse(report["joint_limits_checked"])

    def test_summary_takes_the_worst_level(self) -> None:
        pairs = [{"baseline": {"safety": {"level": "pass"}}}, {"baseline": {"safety": {"level": "review"}}},
                 {"baseline": {"safety": None}}]
        self.assertEqual(safety_summary(pairs), {"baseline": {
            "level": "review", "counts": {"pass": 1, "review": 1, "unavailable": 1}, "trials": 3}})


class PageTests(unittest.TestCase):
    def test_panels_render(self) -> None:
        review = safety_report([row(0.0, [("arm_object", ("left_tip_left", "bowl"), 5.0)])], LIMITS)
        pairs = [{"start": 1, "baseline": {**trial(stage="untouched"), "safety": review}},
                 {"start": 2, "baseline": {**trial("success", "successed"), "safety": safety_report([row(0.0)], LIMITS)}}]
        modes = web_dashboard.failure_mode_panel(pairs)
        self.assertIn("Never reached the object", modes)
        self.assertIn("#1", modes)
        gate = web_dashboard.safety_panel(pairs)
        self.assertIn('<span class="safety review">review</span>', gate)
        self.assertIn("1 pass · 1 review", gate)
        self.assertIn("1 of 2 trials · peak 5.00 N", gate)
        card = web_dashboard.safety_details(pairs[0]["baseline"], None)
        self.assertIn("bowl – left_tip_left", card)
        self.assertIn("peak 5.0 N", card)


@unittest.skipUnless(HAS_SIM, "Dream YAM simulator not installed")
class LimitTests(unittest.TestCase):
    def test_limits_come_from_the_yam_model(self) -> None:
        from dreamlabs.safety import arm_joint_limits

        limits = arm_joint_limits()
        self.assertEqual(len(limits["left"]), 6)
        self.assertEqual(limits["right"][1][0], 0.0)  # joint2's lower limit is the home angle


if __name__ == "__main__":
    unittest.main()
