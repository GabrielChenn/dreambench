"""Self-contained replay page: camera video, a frame timeline with collision lanes, per-frame review, and
(when opened from the workbench) a 3D review restored from each frame's pre-step physics snapshot."""

from html import escape
import json
from pathlib import Path
import re

from .web_theme import WORKBENCH_CSS

CAMERAS = ("top", "left", "right")
CAMERA_LABELS = {"top": "Overhead", "left": "Left wrist", "right": "Right wrist"}
# Keep in step with COLORS in review_scene.py.
CATEGORY_COLORS = {"arm_table": "#ff4040", "arm_arm": "#f24df2", "arm_self": "#b373ff", "arm_object": "#ff9a1a",
                   "grasp_contact": "#4df273", "object_support": "#59d9ff"}
REPLAY_CSS = """
.panel{background:var(--surface);border:1px solid var(--line);border-radius:.9rem;padding:1.2rem;margin-top:1.25rem}
header nav{display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap}
header a{font-weight:650}
.video-wrap{position:relative;background:#080e17;border-radius:.55rem;overflow:hidden}
video{display:block;width:100%;height:auto}
#boxes{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.cameras{display:flex;margin-bottom:.6rem;gap:.4rem}
.cameras span{flex:1;text-align:center;color:var(--muted);font-size:.875rem}
.controls{display:flex;align-items:end;flex-wrap:wrap;gap:.55rem;margin-top:.8rem}
.controls label{display:grid;gap:.35rem;color:var(--muted);font-size:.875rem}
.toggles{display:flex;flex-wrap:wrap;gap:.9rem;margin-top:.6rem;color:var(--muted);font-size:.9rem}
.toggles label{display:flex;gap:.35rem;align-items:center}
input,button,select{font:inherit;border-radius:.45rem;padding:.4rem .65rem}
input[type=number],select{color:var(--ink);background:#0c131e;border:1px solid #40516b}
input[type=number]{width:7rem;font-variant-numeric:tabular-nums}
input[type=checkbox]{padding:0}
button{color:#081d19;background:var(--accent);border:1px solid transparent;font-weight:700;cursor:pointer}
button:disabled{opacity:.55;cursor:default}
.secondary{background:#26364a;color:var(--ink)}
output{display:block;color:var(--muted);margin-top:.5rem;font-size:.9rem;font-variant-numeric:tabular-nums}
.player{display:grid;grid-template-columns:6.5rem 1fr auto auto;align-items:center;gap:.7rem;margin-top:.7rem}
#play{width:2.6rem;height:2.6rem;padding:0;border-radius:50%;font-size:1.05rem;flex:none}
#timeline{width:100%;min-width:6rem;margin:0;accent-color:var(--accent)}
.clock{color:var(--muted);font-size:.9rem;font-variant-numeric:tabular-nums;white-space:nowrap}
.speed{display:flex;gap:.35rem;align-items:center;color:var(--muted);font-size:.875rem}
#events{display:block;width:100%;height:96px;margin-top:.5rem;touch-action:none;background:#0c131e;border:1px solid var(--line);border-radius:.45rem;cursor:pointer}
.review-wrap{position:relative;max-width:60rem;margin:0 auto;background:#080e17;border-radius:.55rem;overflow:hidden;aspect-ratio:4/3;touch-action:none;cursor:grab}
.review-wrap img{display:block;width:100%;height:100%;object-fit:contain;user-select:none;-webkit-user-drag:none}
.review-wrap p{position:absolute;inset:auto .7rem .6rem;margin:0;color:#d3e5f6;font-size:.85rem;text-shadow:0 1px 2px #000}
.legend{display:flex;flex-wrap:wrap;gap:.35rem .9rem;margin-top:.6rem;font-size:.82rem;color:var(--muted)}
.legend span::before{content:"";display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:.35rem;background:var(--c);vertical-align:-.05rem}
.table-wrap{overflow:auto;margin:.5rem 0}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.92rem}
th,td{padding:.35rem .5rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,th.text,td.text{text-align:left}
thead th{color:var(--muted);font-weight:650;font-size:.85rem}
td.bad{color:#ffb2b2}
.chip{display:inline-block;width:.65rem;height:.65rem;border-radius:50%;margin-right:.4rem;vertical-align:-.03rem}
h3{font-size:1rem;margin:1rem 0 .2rem}
pre{background:#0c131e;border:1px solid var(--line);color:#d3e5f6;border-radius:.55rem;padding:1rem;overflow:auto;margin:0;max-height:24rem}
"""

TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DreamBench Replay</title><style>__CSS__</style>
<header><nav aria-label="Workbench"><strong><span>◆</span>DreamBench</strong><a id="back" hidden>Back to results</a></nav></header>
<main><h1>Simulation replay</h1>
<p class="description">Play or step through the run: a 3D review of each frame with its contact timeline, the camera views at the same moment, trajectories, and every contact.</p>
<section class="panel" aria-labelledby="review-title">
<h2 id="review-title">3D review</h2>
<div class="review-wrap" id="review-wrap"><img id="review-image" alt="3D view of this frame restored from its physics snapshot"><p id="review-note">Loading the 3D review…</p></div>
<div class="player">
<button id="play" type="button" aria-label="Play" title="Play / pause (Space)">▶</button>
<input id="timeline" type="range" min="0" max="0" step="1" value="0" aria-label="Frame">
<span id="clock" class="clock">0.00 / 0.00 s</span>
<label class="speed">Speed <select id="speed" aria-label="Playback speed">
<option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option>
</select></label>
</div>
<canvas id="events" aria-label="Contacts per frame; click or drag to jump"></canvas>
<output id="position" aria-live="polite"></output>
<form id="seek-form" class="controls">
<label>Simulation time (s)<input id="seek" type="number" min="0" step="any" required></label>
<button id="jump" type="submit">Jump</button>
<button id="previous-frame" class="secondary" type="button">◀ Previous frame</button>
<button id="next-frame" class="secondary" type="button">Next frame ▶</button>
<button id="previous-collision" class="secondary" type="button">⏮ Previous collision</button>
<button id="next-collision" class="secondary" type="button">Next collision ⏭</button>
</form>
<p class="description" id="lanes-source"></p>
<div class="toggles">
<label>Follow <select id="follow"><option value="">Scene</option></select></label>
<label><input id="show-contacts" type="checkbox" checked>Contacts</label>
<label><input id="show-other" type="checkbox">Resting contacts</label>
<label><input id="show-trails" type="checkbox" checked>Trajectories</label>
<button id="reset-view" class="secondary" type="button">Reset view</button>
</div>
<div class="legend" id="legend"></div>
<p class="description">Play, drag the timeline, or click a contact lane: the 3D review and the camera views move to the same time.
Drag the 3D view to rotate, scroll to zoom. Each frame is restored from the MuJoCo state saved just before its control step.
Trajectories are solid up to this frame and faint after it.</p>
</section>
<section class="panel" aria-labelledby="video-title">
<h2 id="video-title">Camera views</h2>
<div class="cameras">__CAMERA_LABELS__</div>
<div class="video-wrap"><video id="video" src="video.mp4" controls preload="metadata" muted playsinline></video><canvas id="boxes" hidden></canvas></div>
<p class="description" id="detection-note">__DETECTION_NOTE__</p>
</section>
<section class="panel" aria-labelledby="frame-title">
<h2 id="frame-title">Frame review</h2>
<p id="action-context" class="description"></p>
<div><h3>Joints</h3><div class="table-wrap"><table>
<thead><tr><th scope="col">Joint</th><th scope="col">Left target</th><th scope="col">Left measured</th><th scope="col">Left error</th>
<th scope="col">Right target</th><th scope="col">Right measured</th><th scope="col">Right error</th></tr></thead>
<tbody id="action-values"></tbody></table></div></div>
<div><h3>Objects</h3><div class="table-wrap"><table>
<thead><tr><th scope="col">Body</th><th scope="col">x (mm)</th><th scope="col">y (mm)</th><th scope="col">z (mm)</th>
<th scope="col">Speed (mm/s)</th><th scope="col" class="text">Touching</th></tr></thead>
<tbody id="object-values"></tbody></table></div></div>
<h3>Collisions and contacts</h3>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Type</th><th scope="col" class="text">Bodies</th><th scope="col">Normal force (N)</th><th scope="col" class="text">When</th></tr></thead>
<tbody id="contact-values"></tbody></table></div>
<div class="controls"><button id="export-frame" type="button" class="secondary">Download frame JSON</button></div>
<p id="reproduction" class="description"></p>
</section>
<section class="panel" aria-labelledby="log-title">
<h2 id="log-title">Frame telemetry</h2>
<pre id="log"></pre>
</section></main>
<script id="rows" type="application/json">__ROWS__</script>
<script id="detections" type="application/json">__DETECTIONS__</script>
<script id="colors" type="application/json">__COLORS__</script>
<script>
'use strict';
const $ = id => document.getElementById(id);
const rows = JSON.parse($('rows').textContent);
const detections = JSON.parse($('detections').textContent);
const COLORS = JSON.parse($('colors').textContent);
const LABELS = {arm_table: 'Arm–table', arm_arm: 'Arm–arm', arm_self: 'Arm self-collision', arm_object: 'Arm hits object',
                grasp_contact: 'Finger grasp', object_support: 'Object on bowl/plate', apple_bowl: 'Object on bowl/plate'};
const LANES = ['arm_table', 'arm_arm', 'arm_self', 'arm_object', 'grasp_contact', 'object_support'];
const COLLISIONS = ['arm_table', 'arm_arm', 'arm_self', 'arm_object'];
const lane = category => category === 'apple_bowl' ? 'object_support' : category;
const video = $('video'), slider = $('timeline'), strip = $('events'), image = $('review-image');
const jointNames = ['Joint 1 (°)', 'Joint 2 (°)', 'Joint 3 (°)', 'Joint 4 (°)', 'Joint 5 (°)', 'Joint 6 (°)', 'Gripper (0–1)'];
const DEG = 180 / Math.PI;
let current = -1, mediaReady = false, review = null, camera = null, cameraDefault = null;
let renderBusy = false, renderPending = false, imageUrl = null, drag = null;
let clockPlaying = false, clockStart = null;  // playback without a camera video
const PLAYBACK = new Set(['video', 'play']);  // sources that keep playback running
slider.max = Math.max(0, rows.length - 1);

// Where this page is served from decides whether the 3D review is available.
const parts = location.pathname.split('/');
const served = location.protocol.startsWith('http') && parts[1] === 'artifacts';
const recording = served ? parts.slice(2, -1).map(decodeURIComponent).join('/') : null;
if (served) {
  const back = $('back');
  back.href = parts[2]?.startsWith('comparison-') ? '/?' + new URLSearchParams({run: parts[2]}) : '/';
  back.hidden = false;
}
$('legend').innerHTML = LANES.map(c => '<span style="--c:' + COLORS[c] + '">' + LABELS[c] + '</span>').join('')
  + '<span style="--c:#4da6ff">Left gripper path</span><span style="--c:#ff66bf">Right gripper path</span>';

// Contacts per frame: the 3D review's (restored, before the step) when loaded, else the recorded ones (after it).
function frameContacts(index) {
  if (review?.contacts?.[index]) return review.contacts[index];
  return rows[index]?.contact_samples ?? null;
}
function lanesOf(index) {
  const set = new Set();
  for (const contact of frameContacts(index) ?? []) set.add(lane(contact.category));
  return set;
}
function isCollision(index) {
  return (frameContacts(index) ?? []).some(contact => COLLISIONS.includes(contact.category));
}

// Frame axis of the lanes in strip pixels: lined up under the slider thumb's travel.
function stripRange() {
  const box = strip.getBoundingClientRect(), track = slider.getBoundingClientRect(), thumb = 8;
  if (!track.width) return [92, box.width - 4];
  return [track.left - box.left + thumb, track.right - box.left - thumb];
}
function drawStrip() {
  const ratio = window.devicePixelRatio || 1;
  const width = strip.clientWidth, height = strip.clientHeight;
  strip.width = width * ratio; strip.height = height * ratio;
  const ctx = strip.getContext('2d');
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const [label, right] = stripRange();
  const laneHeight = height / LANES.length, span = Math.max(1, rows.length - 1);
  ctx.font = '11px system-ui';
  LANES.forEach((name, i) => {
    ctx.fillStyle = '#a9b8cc';
    ctx.fillText(LABELS[name], 4, i * laneHeight + laneHeight * 0.7);
  });
  const known = rows.some((_, i) => frameContacts(i) !== null);
  if (!known) {
    ctx.fillStyle = '#a9b8cc';
    ctx.fillText('No contact telemetry in this recording.', label + 8, height / 2);
  }
  for (let i = 0; i < rows.length; i++) {
    const x = label + (right - label) * i / span;
    for (const name of lanesOf(i)) {
      const row = LANES.indexOf(name);
      if (row < 0) continue;
      ctx.fillStyle = COLORS[name];
      ctx.fillRect(x, row * laneHeight + 2, Math.max(1, (right - label) / span), laneHeight - 4);
    }
  }
  if (current >= 0) {
    const x = label + (right - label) * current / span;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(x - 1, 0, 2, height);
  }
}
function stripFrame(event, clamp) {
  const [label, right] = stripRange();
  const fraction = (event.clientX - strip.getBoundingClientRect().left - label) / (right - label);
  if (fraction < 0 && !clamp) return null;
  return Math.round(Math.max(0, Math.min(1, fraction)) * (rows.length - 1));
}
let stripDrag = false;
strip.addEventListener('pointerdown', event => {
  const index = stripFrame(event, false);
  if (index === null) return;
  stripDrag = true;
  strip.setPointerCapture(event.pointerId);
  setFrame(index, 'strip');
});
strip.addEventListener('pointermove', event => { if (stripDrag) setFrame(stripFrame(event, true), 'strip'); });
const endStripDrag = () => { stripDrag = false; };
strip.addEventListener('pointerup', endStripDrag);
strip.addEventListener('pointercancel', endStripDrag);
window.addEventListener('resize', drawStrip);

function cell(tr, text, className) {
  const td = tr.insertCell();
  td.textContent = text;
  if (className) td.className = className;
  return td;
}
const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';

function showJoints(row) {
  const body = $('action-values');
  body.replaceChildren();
  const target = row?.commanded_joint_rad_and_gripper, measured = row?.measured_joint_rad_and_gripper;
  jointNames.forEach((name, joint) => {
    const tr = body.insertRow();
    const th = document.createElement('th');
    th.scope = 'row'; th.textContent = name; tr.append(th);
    for (const side of ['left', 'right']) {
      const scale = joint < 6 ? DEG : 1;
      const t = target?.[side]?.[joint] * scale, m = measured?.[side]?.[joint] * scale;
      const error = m - t;
      cell(tr, fmt(t));
      cell(tr, fmt(m));
      cell(tr, fmt(error), Number.isFinite(error) && Math.abs(error) > (joint < 6 ? 5 : 0.2) ? 'bad' : '');
    }
  });
}

function showObjects(index) {
  const body = $('object-values');
  body.replaceChildren();
  const row = rows[index], previous = rows[index - 1];
  for (const [name, xyz] of Object.entries(row?.object_xyz_m ?? {})) {
    const tr = body.insertRow();
    cell(tr, name.replace('_', ' '), 'text');
    xyz.forEach(value => cell(tr, fmt(value * 1000, 1)));
    const before = previous?.object_xyz_m?.[name];
    const speed = before ? Math.hypot(...xyz.map((v, k) => v - before[k])) * 1000 / 0.05 : NaN;
    cell(tr, fmt(speed, 1));
    const touching = row[name + '_contacts'];
    cell(tr, Array.isArray(touching) ? (touching.join(', ') || '—') : 'not recorded', 'text');
  }
}

function showContacts(index) {
  const body = $('contact-values');
  body.replaceChildren();
  const groups = [];
  if (review?.contacts?.[index]) groups.push(['Before this step (3D)', review.contacts[index]]);
  if (Array.isArray(rows[index]?.contact_samples)) groups.push(['After this step (recorded)', rows[index].contact_samples]);
  let count = 0;
  for (const [when, contacts] of groups) {
    for (const contact of [...contacts].sort((a, b) => LANES.indexOf(lane(a.category)) - LANES.indexOf(lane(b.category)))) {
      const tr = body.insertRow();
      const type = tr.insertCell();
      type.className = 'text';
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.style.background = COLORS[lane(contact.category)] ?? '#bbb';
      type.append(chip, LABELS[contact.category] ?? contact.category);
      cell(tr, contact.bodies.join(' – '), 'text');
      cell(tr, fmt(contact.normal_force_n, 2));
      cell(tr, when, 'text');
      count++;
    }
  }
  if (!count) {
    const tr = body.insertRow();
    const td = cell(tr, groups.length ? 'No tracked contacts in this frame.' : 'Contact telemetry unavailable for this frame.', 'text');
    td.colSpan = 4;
  }
}

function frameText(index) {
  const row = rows[index];
  let text = row.sim_seconds.toFixed(2) + ' s · frame ' + index + ' of ' + (rows.length - 1);
  const names = [...lanesOf(index)].filter(name => COLLISIONS.includes(name)).map(name => LABELS[name]);
  text += names.length ? ' · Collision: ' + names.join(', ') : ' · No collision';
  const detection = detections?.[index];
  if (detections) {
    text += ' · YOLO (overhead / left / right): ' + (detection?.ran
      ? ['top', 'left', 'right'].map(camera => detection.cameras[camera].length).join(' / ') : 'not run');
  }
  return text;
}

function setFrame(index, source) {
  if (!rows.length) return;
  index = Math.max(0, Math.min(rows.length - 1, index));
  if (!PLAYBACK.has(source)) stopClock();  // stepping or scrubbing pauses playback
  if (source !== 'video' && mediaReady) {
    if (!PLAYBACK.has(source)) video.pause();
    video.currentTime = rows[index].sim_seconds + 0.001;
  }
  if (index === current) { draw(detections?.[index]); return; }
  current = index;
  const row = rows[index];
  slider.value = index;
  $('seek').value = row.sim_seconds.toFixed(2);
  $('clock').textContent = row.sim_seconds.toFixed(2) + ' / ' + rows[rows.length - 1].sim_seconds.toFixed(2) + ' s';
  $('position').textContent = frameText(index);
  $('previous-frame').disabled = index <= 0;
  $('next-frame').disabled = index >= rows.length - 1;
  $('action-context').textContent = 'Targets are sent before the step. Measured values, objects and recorded contacts are sampled after it. '
    + 'Errors above 5° (or 0.2 gripper) are highlighted.';
  $('reproduction').textContent = row.physics_before_step
    ? 'Physics snapshot available. Download the frame and run python record_replay.py --reproduce-frame <downloaded.json>.'
    : 'This recording has joint telemetry only; physics snapshots are in newer recordings.';
  showJoints(row);
  showObjects(index);
  showContacts(index);
  const {physics_before_step, ...telemetry} = row;
  if (physics_before_step) telemetry.physics_snapshot_available = true;
  if (detections?.[index]) telemetry.detection = detections[index];
  if (review?.tips?.[index]) telemetry.gripper_tip_xyz_m_before_step = review.tips[index];
  $('log').textContent = JSON.stringify(telemetry, null, 2);
  draw(detections?.[index]);
  drawStrip();
  requestRender();
}

function seekCollision(direction) {
  for (let i = current + direction; i >= 0 && i < rows.length; i += direction) {
    // Land on the first frame of each collision.
    if (isCollision(i) && !isCollision(i - 1)) { setFrame(i, 'button'); return; }
  }
  $('position').textContent = frameText(current) + (direction > 0 ? ' · No later collision' : ' · No earlier collision');
}

// 3D review ------------------------------------------------------------------------------------
function followTarget(index) {
  const name = $('follow').value;
  if (!name) return null;
  if (name === 'left' || name === 'right') return review?.tips?.[index]?.[name] ?? null;
  return rows[index]?.object_xyz_m?.[name] ?? null;
}
function requestRender() {
  if (!review || current < 0) return;
  if (renderBusy) { renderPending = true; return; }
  renderBusy = true;
  const lookat = followTarget(current) ?? camera.lookat;
  const params = new URLSearchParams({
    recording, frame: current, az: camera.azimuth.toFixed(1), el: camera.elevation.toFixed(1),
    dist: camera.distance.toFixed(3), lx: lookat[0].toFixed(4), ly: lookat[1].toFixed(4), lz: lookat[2].toFixed(4),
    contacts: $('show-contacts').checked ? 1 : 0, other: $('show-other').checked ? 1 : 0,
    trails: $('show-trails').checked ? 1 : 0,
  });
  const frame = current;
  fetch('/api/review/frame.jpg?' + params, {cache: 'no-store'}).then(async response => {
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'The 3D review failed.');
    const url = URL.createObjectURL(await response.blob());
    image.src = url;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = url;
    $('review-note').textContent = 'Before the step of frame ' + frame + ' · ' + rows[frame].sim_seconds.toFixed(2) + ' s';
  }).catch(error => {
    $('review-note').textContent = error.message;
  }).finally(() => {
    renderBusy = false;
    if (renderPending) { renderPending = false; requestRender(); }
  });
}

async function loadReview() {
  if (!served) {
    $('review-note').textContent = 'Open this replay from the DreamBench workbench to review it in 3D.';
    return;
  }
  try {
    const response = await fetch('/api/review/summary?' + new URLSearchParams({recording}), {cache: 'no-store'});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'The 3D review is unavailable.');
    review = body;
    cameraDefault = {...review.camera, lookat: [...review.camera.lookat]};
    camera = {...cameraDefault, lookat: [...cameraDefault.lookat]};
    const options = [['left', 'Left gripper'], ['right', 'Right gripper'],
      ...Object.keys(rows[0]?.object_xyz_m ?? {}).map(name => [name, name.replace('_', ' ')])];
    for (const [value, label] of options) $('follow').append(new Option(label, value));
    $('lanes-source').textContent = 'Lanes show contacts found in each restored 3D state (' + review.restorable + ' of '
      + review.frames + ' frames restorable). Click a lane to jump there.';
    $('position').textContent = frameText(current);
    showContacts(current);
    drawStrip();
    requestRender();
  } catch (error) {
    $('review-note').textContent = error.message;
    $('lanes-source').textContent = 'Lanes show the contacts recorded after each step.';
  }
}

const wrap = $('review-wrap');
wrap.addEventListener('pointerdown', event => {
  if (!camera) return;
  drag = {x: event.clientX, y: event.clientY};
  wrap.setPointerCapture(event.pointerId);
  wrap.style.cursor = 'grabbing';
});
wrap.addEventListener('pointermove', event => {
  if (!drag) return;
  camera.azimuth -= (event.clientX - drag.x) * 0.35;
  camera.elevation = Math.max(-89, Math.min(89, camera.elevation - (event.clientY - drag.y) * 0.3));
  drag = {x: event.clientX, y: event.clientY};
  requestRender();
});
const endDrag = () => { drag = null; wrap.style.cursor = 'grab'; };
wrap.addEventListener('pointerup', endDrag);
wrap.addEventListener('pointercancel', endDrag);
wrap.addEventListener('wheel', event => {
  if (!camera) return;
  event.preventDefault();
  camera.distance = Math.max(0.15, Math.min(6, camera.distance * Math.exp(event.deltaY * 0.0012)));
  requestRender();
}, {passive: false});
$('follow').addEventListener('change', () => {
  if (camera && $('follow').value) camera.distance = Math.min(camera.distance, 0.6);
  requestRender();
});
for (const id of ['show-contacts', 'show-other', 'show-trails']) $(id).addEventListener('change', requestRender);
$('reset-view').addEventListener('click', () => {
  if (!cameraDefault) return;
  camera = {...cameraDefault, lookat: [...cameraDefault.lookat]};
  $('follow').value = '';
  requestRender();
});

// Camera video and YOLO boxes ------------------------------------------------------------------
const canvas = $('boxes');
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
  for (const [cameraIndex, name] of ['top', 'left', 'right'].entries()) {
    for (const object of record.cameras[name]) {
      const [x1, y1, x2, y2] = object.bbox_xyxy_px;
      const x = cameraIndex * width + x1;
      const color = {apple: '#ff6363', orange: '#ffab45', banana: '#fff16b', bowl: '#67dbff'}[object.label] ?? '#a7ff83';
      const label = object.confidence == null ? object.label : object.label + ' ' + Math.round(object.confidence * 100) + '%';
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
const frameAt = seconds => Math.floor(seconds * 20 + 1e-6);
video.addEventListener('loadedmetadata', () => {
  const resume = clockPlaying;  // hand playback over from the wall clock to the video
  stopClock();
  mediaReady = true;
  const requested = Number(new URLSearchParams(location.search).get('t'));
  if (rows.length && new URLSearchParams(location.search).has('t') && Number.isFinite(requested)) {
    setFrame(frameAt(Math.max(0, requested)), 'query');
  } else {
    setFrame(Math.max(current, 0), 'sync');  // keep the frame chosen while the video loaded
  }
  if (resume) play();
});
video.addEventListener('timeupdate', () => setFrame(frameAt(video.currentTime), 'video'));
video.addEventListener('seeked', () => setFrame(frameAt(video.currentTime), 'video'));
if (video.requestVideoFrameCallback) {
  const onFrame = (_, metadata) => { setFrame(frameAt(metadata.mediaTime), 'video'); video.requestVideoFrameCallback(onFrame); };
  video.requestVideoFrameCallback(onFrame);
}

// Playback ------------------------------------------------------------------------------------
// With the camera video loaded, the video is the clock: its frames drive the timeline and the 3D review.
// Without it, a wall clock steps through the recorded frames at their simulation times.
const speed = () => Number($('speed').value) || 1;
const isPlaying = () => clockPlaying || (mediaReady && !video.paused && !video.ended);
function showPlayState() {
  const playing = isPlaying();
  $('play').textContent = playing ? '❚❚' : '▶';
  $('play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
}
function stopClock() {
  if (!clockPlaying) return;
  clockPlaying = false;
  showPlayState();
}
function tick(now) {
  if (!clockPlaying) return;
  const last = rows.length - 1;
  const seconds = clockStart.sim + (now - clockStart.wall) / 1000 * speed();
  let index = current;
  while (index < last && rows[index + 1].sim_seconds <= seconds) index++;
  setFrame(index, 'play');
  if (index >= last) { stopClock(); return; }
  requestAnimationFrame(tick);
}
function play() {
  if (!rows.length) return;
  if (current >= rows.length - 1) setFrame(0, 'button');  // replay from the start
  if (mediaReady) {
    video.playbackRate = speed();
    video.play().catch(error => { $('review-note').textContent = 'Playback failed: ' + error.message; });
  } else {
    clockPlaying = true;
    clockStart = {wall: performance.now(), sim: rows[current].sim_seconds};
    requestAnimationFrame(tick);
  }
  showPlayState();
}
function pause() {
  stopClock();
  if (mediaReady) video.pause();
  showPlayState();
}
$('play').onclick = () => isPlaying() ? pause() : play();
$('speed').addEventListener('change', () => {
  video.playbackRate = speed();
  if (clockPlaying) clockStart = {wall: performance.now(), sim: rows[current].sim_seconds};
});
for (const name of ['play', 'pause', 'ended']) video.addEventListener(name, showPlayState);

// Controls ------------------------------------------------------------------------------------
slider.addEventListener('input', () => setFrame(Number(slider.value), 'slider'));
$('seek-form').addEventListener('submit', event => {
  event.preventDefault();
  if (Number.isFinite($('seek').valueAsNumber)) setFrame(frameAt($('seek').valueAsNumber), 'seek');
});
$('previous-frame').onclick = () => setFrame(current - 1, 'button');
$('next-frame').onclick = () => setFrame(current + 1, 'button');
$('previous-collision').onclick = () => seekCollision(-1);
$('next-collision').onclick = () => seekCollision(1);
document.addEventListener('keydown', event => {
  if (event.target.matches('input[type=number], select, textarea')) return;
  if (event.key === ' ' && !event.target.matches('button, video')) {
    event.preventDefault();
    isPlaying() ? pause() : play();
  }
  if (event.key === 'ArrowLeft') { event.preventDefault(); setFrame(current - 1, 'key'); }
  if (event.key === 'ArrowRight') { event.preventDefault(); setFrame(current + 1, 'key'); }
});
$('export-frame').onclick = () => {
  const row = rows[current];
  if (!row) return;
  const extra = {};
  if (detections?.[current]) extra.detection = detections[current];
  if (review?.contacts?.[current]) extra.contacts_before_step = review.contacts[current];
  const url = URL.createObjectURL(new Blob([JSON.stringify({...row, ...extra}, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'dreambench-frame-' + row.frame + '.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

if (rows.length) {
  const requested = Number(new URLSearchParams(location.search).get('t'));
  setFrame(Number.isFinite(requested) ? frameAt(Math.max(0, requested)) : 0, 'start');
} else {
  $('position').textContent = 'No frames';
}
loadReview();
</script></html>"""


def _json(value) -> str:
    return json.dumps(value, allow_nan=False).replace("<", "\\u003c")


def replay_html(rows: list[dict], *, detections: list[dict] | None = None,
                cameras: tuple[str, ...] = CAMERAS, detection_note: str | None = None) -> str:
    """Build the replay page. It needs only its sibling video.mp4; the 3D review needs the workbench.
    With detections, the YOLO boxes are drawn on the camera views; detection_note says where they came from or why not."""
    labels = "".join(f"<span>{escape(CAMERA_LABELS.get(camera, camera))}</span>" for camera in cameras)
    if detection_note is None:
        detection_note = "YOLO boxes are drawn on the camera views." if detections else ""
    replacements = {"__CSS__": "\n" + WORKBENCH_CSS + REPLAY_CSS, "__CAMERA_LABELS__": labels,
                    "__DETECTION_NOTE__": escape(detection_note),
                    "__COLORS__": _json(CATEGORY_COLORS), "__DETECTIONS__": _json(detections), "__ROWS__": _json(rows)}
    # One pass, so recorded text that happens to contain a marker is never substituted.
    return re.sub("|".join(map(re.escape, replacements)), lambda match: replacements[match.group(0)], TEMPLATE)


def write_replay(path: Path, rows: list[dict], *, detections: list[dict] | None = None,
                 cameras: tuple[str, ...] = CAMERAS) -> None:
    path.write_text(replay_html(rows, detections=detections, cameras=cameras), encoding="utf-8")
