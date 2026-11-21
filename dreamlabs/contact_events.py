"""Classify sampled MuJoCo contacts and summarize their replay evidence."""


from __future__ import annotations
from math import isclose

FRUITS = {"apple", "orange", "banana", "water_bottle"}  # the scene's movable objects
SUPPORTS = {"bowl", "plate"}
EVENTS = ("grasp_contact", "arm_table", "arm_arm")
# Every category contact_category can return, in display order. The first four are collisions.
CATEGORIES = ("arm_table", "arm_arm", "arm_self", "arm_object", "grasp_contact", "apple_bowl", "object_support")
COLLISIONS = ("arm_table", "arm_arm", "arm_self", "arm_object")
TIPS = ("left_tip_", "right_tip_")


def arm(body: str) -> str | None:
    side = body.partition("_")[0]
    return side if side in ("left", "right") else None


def contact_category(geom_a: str, body_a: str, geom_b: str, body_b: str) -> str | None:
    """Classify one MuJoCo contact by the bodies it joins. Object-on-table rest contacts return None."""
    side_a, side_b = arm(body_a), arm(body_b)
    if side_a and side_b:
        return "arm_arm" if side_a != side_b else "arm_self"
    if (geom_a == "table" and side_b) or (geom_b == "table" and side_a):
        return "arm_table"
    if ((body_a in FRUITS and body_b.startswith(TIPS))
            or (body_b in FRUITS and body_a.startswith(TIPS))):
        return "grasp_contact"
    if {body_a, body_b} == {"apple", "bowl"}:
        return "apple_bowl"
    if (side_a or side_b) and "world" not in (body_a, body_b):
        return "arm_object"  # an arm link (not a fingertip on an object) hitting an object or container
    if (body_a in FRUITS and body_b in SUPPORTS) or (body_b in FRUITS and body_a in SUPPORTS):
        return "object_support"
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
