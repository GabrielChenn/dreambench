"""Infer the judged task from a Dream-1 prompt, in English or Chinese. Rule-based; calls no LLM.

"put orange into the bowl" and "把橙子放进碗里" both name one movable object and one support of the
scene, so they map to orange-in-bowl and its scorer. Ported from dreambench_bk/judge_dashboard.py.
"""

from __future__ import annotations

import re

from .scene_inventory import Inventory, load_inventory
from .scoring import GENERIC, TaskSpec, phrase, task_for

# Extra words for scene bodies, mapped to the simulator's body names. Each body's own name (with
# spaces for underscores) and its plural are added automatically, so a body added to the scene can be
# named in English without editing this table. Words for bodies the scene lacks are ignored.
ALIASES = {
    "苹果": "apple",
    "橙子": "orange", "橘子": "orange", "桔子": "orange",
    "香蕉": "banana",
    "bottle": "water_bottle", "bottles": "water_bottle", "水瓶": "water_bottle", "瓶子": "water_bottle",
    "矿泉水": "water_bottle",
    "碗": "bowl", "dish": "plate", "盘子": "plate", "盘": "plate",
}
PLACE_VERBS = ("put", "place", "move", "drop", "set", "transfer", "pick", "放", "移", "拿", "搬", "装")
MAX_PROMPT = 500
GENERIC_NOTE = ("Generic placement rule: this pair was not calibrated in MuJoCo. The verdict uses contacts "
                "only, so check the replay before trusting it.")


def vocabulary(names: tuple[str, ...]) -> dict[str, str]:
    """Words that name each body: its name, its name with spaces, plurals, and known aliases."""
    words: dict[str, str] = {}
    for name in names:
        spoken = name.replace("_", " ")
        for word in (name, spoken, spoken + "s", spoken + "es"):
            words[word] = name
    words.update({word: name for word, name in ALIASES.items() if name in names})
    return words


def find_terms(text: str, words: dict[str, str]) -> list[str]:
    """Return the canonical names mentioned in text, in order, without duplicates."""
    found: list[tuple[int, str]] = []
    lower = text.lower()
    for word, name in words.items():
        pattern = re.escape(word) if not word.isascii() else rf"\b{re.escape(word)}\b"
        match = re.search(pattern, lower)
        if match:
            found.append((match.start(), name))
    ordered: list[str] = []
    for _, name in sorted(found):
        if name not in ordered:
            ordered.append(name)
    return ordered


def parse_prompt(text: str, scene: Inventory | None = None) -> TaskSpec:
    """Return the task a prompt asks for, or raise ValueError with a message for the page."""
    scene = scene or load_inventory()
    text = " ".join(str(text).split())
    if not text:
        raise ValueError("Describe the task, e.g. “put the apple into the bowl” or “把苹果放进碗里”.")
    if len(text) > MAX_PROMPT:
        raise ValueError(f"Keep the prompt under {MAX_PROMPT} characters.")
    objects = find_terms(text, vocabulary(scene.objects))
    targets = find_terms(text, vocabulary(scene.supports))
    if not objects:
        raise ValueError("Name an object the scene has: " + ", ".join(map(phrase, scene.objects)) + ".")
    if len(objects) > 1:
        raise ValueError(f"Name one object per run. Found: {', '.join(map(phrase, objects))}.")
    if not targets:
        raise ValueError("Name where it goes. The scene has: " + ", ".join(map(phrase, scene.supports)) + ".")
    if len(targets) > 1:
        raise ValueError(f"Name one destination per run. Found: {', '.join(map(phrase, targets))}.")
    if not any(verb in text.lower() for verb in PLACE_VERBS):
        raise ValueError("Say what to do with it, e.g. “put … into …” or “把…放进…”.")
    return task_for(objects[0], targets[0])


def describe(spec: TaskSpec) -> dict:
    """The task fields the Web page shows and the CLI configuration stores."""
    return {"task_id": spec.task_id, "object": spec.object, "target": spec.target, "scorer": spec.scorer,
            "tier": spec.tier, "success_condition": spec.success_condition,
            "note": GENERIC_NOTE if spec.tier == GENERIC else None}
