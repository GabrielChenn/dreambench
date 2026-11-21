"""Classify sampled MuJoCo contacts and summarize their replay evidence."""


from __future__ import annotations
from math import isclose

FRUITS = {"apple", "orange", "banana"}
EVENTS = ("grasp_contact", "arm_table", "arm_arm")


def arm(body: str) -> str | None:
    side = body.partition("_")[0]
    return side if side in ("left", "right") else None


def contact_category(geom_a: str, body_a: str, geom_b: str, body_b: str) -> str | None:
    side_a, side_b = arm(body_a), arm(body_b)
    if side_a and side_b and side_a != side_b:
        return "arm_arm"
    if (geom_a == "table" and side_b) or (geom_b == "table" and side_a):
        return "arm_table"
    if ((body_a in FRUITS and body_b.startswith(("left_tip_", "right_tip_")))
            or (body_b in FRUITS and body_a.startswith(("left_tip_", "right_tip_")))):
        return "grasp_contact"
    if {body_a, body_b} == {"apple", "bowl"}:
        return "apple_bowl"
    return None


def sample_contacts(model, data) -> list[dict]:
    import mujoco
    import numpy as np

    force = np.empty(6)
    entries = []
    for index, contact in enumerate(data.contact[:data.ncon]):
        if contact.efc_address < 0:
            continue
        geom_a, geom_b = map(int, contact.geom)
        body_a, body_b = [model.body(int(model.geom_bodyid[geom])).name for geom in (geom_a, geom_b)]
        name_a, name_b = model.geom(geom_a).name, model.geom(geom_b).name
        category = contact_category(name_a, body_a, name_b, body_b)
        if category is None:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        normal = float(force[0])
        if normal > 0:
            entries.append({
                "category": category, "bodies": [body_a, body_b],
                "geom_ids": [geom_a, geom_b], "normal_force_n": normal,
            })
    return entries


def contact_summary(rows: list[dict]) -> dict:
    counts = {key: {"events": 0, "first_sim_s": None} for key in EVENTS}
    last_seen = {}
    peak = None
    sampled = 0
    for row in rows:
        contacts = row.get("contact_samples")
        if not isinstance(contacts, list):
            continue
        sampled += 1
        timestamp = row["sim_seconds"]
        categories = {contact["category"] for contact in contacts}
        for key in categories.intersection(EVENTS):
            metric = counts[key]
            if key not in last_seen or not isclose(timestamp - last_seen[key], 0.05, abs_tol=1e-6):
                metric["events"] += 1
                if metric["first_sim_s"] is None:
                    metric["first_sim_s"] = timestamp
            last_seen[key] = timestamp
        # Sum normal-force magnitudes of only the apple-to-bowl contacts at this sampled instant.
        if any(arm(body) for body in row.get("apple_contacts", ())):
            continue
        bowl_force = sum(contact["normal_force_n"] for contact in contacts if contact["category"] == "apple_bowl")
        if bowl_force > 0 and (peak is None or bowl_force > peak["normal_force_n"]):
            peak = {"normal_force_n": bowl_force, "sim_s": timestamp}
    return {"sampled_frames": sampled, "events": counts, "released_apple_bowl_peak": peak}
