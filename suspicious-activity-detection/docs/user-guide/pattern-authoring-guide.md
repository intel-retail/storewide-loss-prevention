# Pattern Authoring Guide

## How to Define Pose-Based Activity Patterns

This guide walks you through defining new behavioral patterns in
`patterns.yaml`.  No code changes are required — the generic phase engine
evaluates any pattern you define.

---

## Prerequisites

Before writing a pattern you need:

1. **A clear description of the physical behavior** — what body parts move,
   where they start, where they end.
2. **Sample video or keypoint data** to verify thresholds.
3. **The alert type** this pattern should trigger (one of:
   `CONCEALMENT`, `LOITERING`, `ZONE_VIOLATION`, `REPEATED_VISIT`,
   `CHECKOUT_BYPASS`).

---

## Step 1 — Break the Behavior Into Temporal Phases

Every pattern is a **time-ordered sequence of body postures**.  Break
the behavior into 2–4 distinct phases that happen in order.

**Example — "Shelf-to-waist concealment":**

| Phase | What Happens | Duration |
|-------|-------------|----------|
| 1. `hand_raised` | Hand is above waist level (reaching to shelf) | ≥ 2 frames |
| 2. `hand_at_waist` | Hand drops near waist/pocket area | ≥ 2 frames |

**Example — "Body-turn concealment":**

| Phase | What Happens | Duration |
|-------|-------------|----------|
| 1. `facing_with_hand_raised` | Shoulders wide (facing camera) + hand up | ≥ 2 frames |
| 2. `turned_with_hand_at_waist` | Shoulders narrow (turned away) + hand at waist | ≥ 2 frames |

> **Tip:** Fewer phases = more robust detection.  Only add a phase if
> there is a clear, observable posture change between stages.

---

## Step 2 — Identify the Keypoints Involved

The pose model provides **17 COCO keypoints**:

```
              nose
         left_eye  right_eye
         left_ear  right_ear
    left_shoulder  right_shoulder
       left_elbow  right_elbow
        left_wrist  right_wrist
          left_hip  right_hip
         left_knee  right_knee
        left_ankle  right_ankle
```

**Virtual reference points** (computed from keypoints):

| Name | Definition |
|------|-----------|
| `waist_midpoint` | Midpoint of `left_hip` and `right_hip` |
| `chest_midpoint` | Midpoint of `left_shoulder` and `right_shoulder` |
| `torso_center` | Midpoint of `chest_midpoint` and `waist_midpoint` |
| `head_center` | Midpoint of `left_ear` and `right_ear` |

For each phase, decide:
- **Subject:** Which keypoint are you testing? (e.g., `wrist`, `nose`)
- **Reference:** What are you comparing it to? (e.g., `waist_midpoint`, `right_shoulder`)

---

## Step 3 — Choose the Relation

Ten relations are available:

| Relation | Meaning | Needs Threshold? |
|----------|---------|-----------------|
| `above` | Subject is higher in image (subject.y < reference.y) | No |
| `below` | Subject is lower in image (subject.y > reference.y) | No |
| `left_of` | Subject is left of reference (subject.x < reference.x) | No |
| `right_of` | Subject is right of reference (subject.x > reference.x) | No |
| `near` | Distance < threshold × torso_length | Yes |
| `far` | Distance ≥ threshold × torso_length | Yes |
| `moving_fast` | Velocity > threshold × torso_length/frame | Yes |
| `stationary` | Velocity < threshold × torso_length/frame | Yes |
| `bent` | Angle at subject (vertex) is within [min_angle, max_angle] | Yes |
| `straight` | Angle at subject > 150° (default) | Optional |

**Negation:** Prefix any relation with `not_` to invert it:
- `not_near` — distance ≥ threshold × torso_length
- `not_above` — subject is NOT above reference

**Understanding image coordinates:**
- Y-axis: **0 = top** of image, **1 = bottom**.  So `above` means
  smaller y-value.
- Torso length = distance from `chest_midpoint` to `waist_midpoint`.
  All `near`/`far`/velocity thresholds are **fractions of torso length**,
  making them scale-invariant across camera distances.

**Choosing threshold values:**

| Threshold | Meaning |
|-----------|---------|
| 0.3 | Very close (within 30% of torso length) |
| 0.5 | Moderate proximity |
| 0.6 | Default for "near waist" |
| 0.8 | Loosely nearby |
| 1.0+ | Can be far apart |

> **Tip:** Start with a generous threshold (e.g., 0.6–0.8) and tighten
> it after testing on real video to reduce false positives.

---

## Step 4 — Decide per_side Behavior

Set `per_side: true` when:
- The pattern involves one hand/leg but you don't know which one.
- You write `wrist` (short name) and the engine will automatically try
  `left_wrist` then `right_wrist`.

Set `per_side: false` when:
- You need both sides simultaneously (e.g., two-hand shield).
- You use explicit full names like `left_wrist`, `right_wrist`.

**Short names that expand with per_side:**
`wrist`, `elbow`, `shoulder`, `hip`, `knee`, `ankle`, `eye`, `ear`

**Names that never expand** (always used as-is):
`nose`, `left_wrist`, `right_shoulder`, `waist_midpoint`, `chest_midpoint`, etc.

---

## Step 5 — Write the YAML

Use this template:

```yaml
patterns:
  my_new_pattern:                        # unique identifier (snake_case)
    description: "One-line description"  # human-readable
    enabled: false                       # set true when ready for production
    alert_type: CONCEALMENT              # one of the 5 supported alert types

    pose:
      per_side: true                     # try left/right independently
      phases:
        - name: phase_one_name           # descriptive label
          min_frames: 2                  # at least 2 frames must match
          conditions:
            - subject: wrist             # keypoint to test
              relation: above            # spatial relation
              reference: waist_midpoint  # comparison target

        - name: phase_two_name
          min_frames: 2
          conditions:
            - subject: wrist
              relation: near
              reference: waist_midpoint
              threshold: 0.6             # required for near/far/velocity

    vlm:
      enabled: true
      num_frames: 4                      # frames sampled for VLM
      prompt: |
        <your VLM prompt here>
      response_fields:
        - suspicious    # bool — required
        - confidence    # float — required
        - reasoning     # str — required
```

---

## Step 6 — Multiple Conditions per Phase

By default, all conditions within a phase are **AND-ed** — every condition
must be true for a frame to match that phase.

```yaml
phases:
  - name: facing_with_hand_raised
    min_frames: 2
    conditions:
      # Condition 1: hand is above waist
      - subject: wrist
        relation: above
        reference: waist_midpoint
      # Condition 2: shoulders are wide apart (person facing camera)
      - subject: left_shoulder
        relation: far
        reference: right_shoulder
        threshold: 0.8
```

A frame matches this phase only when **both** the hand is raised
**and** the shoulders are wide.

### OR Logic

Set `match: any` on the phase to use OR logic:

```yaml
phases:
  - name: hand_near_body
    min_frames: 2
    match: any                    # frame matches if ANY condition is true
    conditions:
      - subject: wrist
        relation: near
        reference: waist_midpoint
        threshold: 0.6
      - subject: wrist
        relation: near
        reference: chest_midpoint
        threshold: 0.5
```

---

## Step 7 — Use Angle Detection

For detecting bent/straight joints, `subject` is the **vertex** (the joint)
and `reference` is a **list of two endpoints** forming the angle:

```yaml
- subject: elbow                     # vertex where angle is measured
  relation: bent
  reference: [shoulder, wrist]       # the two arms of the angle
  min_angle: 30
  max_angle: 120
```

This checks: "Is the elbow bent between 30° and 120°?"

For `straight`, defaults are min_angle=150, max_angle=180 (can override).

---

## Step 8 — Use Velocity Detection

For detecting movement speed:

```yaml
# Fast hand motion
- subject: wrist
  relation: moving_fast
  threshold: 0.3                # > 30% of torso length per frame

# Person standing still
- subject: left_hip
  relation: stationary
  threshold: 0.05               # < 5% of torso length per frame
```

---

## Step 9 — Window Mode (Stationary Patterns)

For patterns that don't have temporal phases (e.g., loitering), add
`window_size` to evaluate conditions over a sliding window:

```yaml
pose:
  per_side: false
  window_size: 15               # evaluate over 15-frame windows
  phases:
    - name: stationary
      min_frames: 15            # all 15 frames must match
      conditions:
        - subject: left_hip
          relation: stationary
          threshold: 0.05
```

Without `window_size`, ordered phases use sliding split (default).

---

## Step 10 — Write the VLM Prompt

The VLM prompt is sent to the vision-language model after the pose
pattern matches.  It provides visual confirmation to reduce false
positives.

**Prompt structure:**

1. **Role:** "You are a retail loss-prevention analyst..."
2. **Task:** Describe exactly what behavior to look for.
3. **Evidence:** List 3–5 specific visual cues to look for.
4. **Counter-evidence:** List things that should lower confidence.
5. **Calibration:** Define confidence ranges.
6. **Response format:** Always end with the exact JSON format.

**Required VLM response fields:**

```yaml
response_fields:
  - suspicious    # bool  — is the behavior suspicious?
  - confidence    # float — 0.0 to 1.0
  - reasoning     # str   — one-sentence explanation
```

> **Tip:** Be specific in the prompt.  "Hand moves toward pocket" is
> better than "suspicious movement".  The VLM uses the frames and your
> prompt together.

---

## Step 11 — Test the Pattern

### Quick validation — load and check structure

```bash
python3 -c "
import yaml
with open('behavioral-analysis/config/patterns.yaml') as f:
    cfg = yaml.safe_load(f)
for name, p in cfg['patterns'].items():
    phases = p.get('pose', {}).get('phases', [])
    print(f'{name}: {len(phases)} phases, enabled={p.get(\"enabled\", False)}')
"
```

### Run unit tests

```bash
python3 -m pytest behavioral-analysis/tests/test_pose_analyzer.py -v
```

---

## Step 12 — Enable for Production

1. Set `enabled: true` in `patterns.yaml`.
2. If using Docker, ensure `patterns.yaml` is volume-mounted so changes
   don't require a rebuild:
   ```yaml
   volumes:
     - ./config/patterns.yaml:/app/config/patterns.yaml:ro
   ```
3. Restart the `behavioral-analysis` container.

---

## Complete Examples

### Example A — Shelf-to-Waist (Concealment)

**Behavior:** Hand reaches up to shelf, then drops to waist/pocket.

```yaml
shelf_to_waist:
  description: "Hand moves from shelf level to waist/pocket area"
  enabled: true
  alert_type: CONCEALMENT
  pose:
    per_side: true
    phases:
      - name: hand_raised
        min_frames: 2
        conditions:
          - subject: wrist
            relation: above
            reference: waist_midpoint
      - name: hand_at_waist
        min_frames: 2
        conditions:
          - subject: wrist
            relation: near
            reference: waist_midpoint
            threshold: 0.6
```

### Example B — Loitering (Stationary Person)

**Behavior:** Person remains still for extended period.

```yaml
loitering:
  description: "Person remains stationary for too long"
  enabled: true
  alert_type: LOITERING
  pose:
    per_side: false
    window_size: 15
    phases:
      - name: stationary
        min_frames: 15
        conditions:
          - subject: left_hip
            relation: stationary
            threshold: 0.05
```

### Example C — Quick Grab

**Behavior:** Fast hand motion from shelf, then hand settles near body.

```yaml
quick_grab:
  description: "Fast hand motion from shelf toward body"
  enabled: false
  alert_type: CONCEALMENT
  pose:
    per_side: true
    phases:
      - name: fast_reach
        min_frames: 2
        conditions:
          - subject: wrist
            relation: moving_fast
            threshold: 0.3
      - name: at_body
        min_frames: 2
        conditions:
          - subject: wrist
            relation: near
            reference: waist_midpoint
            threshold: 0.5
```

### Example D — Crouching (Bent Knee)

**Behavior:** Person crouches — knee bent acutely.

```yaml
crouching:
  description: "Person crouches or bends significantly"
  enabled: false
  alert_type: ZONE_VIOLATION
  pose:
    per_side: true
    window_size: 8
    phases:
      - name: knee_bent
        min_frames: 4
        conditions:
          - subject: knee
            relation: bent
            reference: [hip, ankle]
            min_angle: 30
            max_angle: 120
```

### Example E — Body Turn + Concealment

**Behavior:** Person faces camera, then turns away while hand drops to waist.

```yaml
body_turn_concealment:
  description: "Turns away from camera while moving hand to waist"
  enabled: false
  alert_type: CONCEALMENT
  pose:
    per_side: true
    phases:
      - name: facing_with_hand_up
        min_frames: 2
        conditions:
          - subject: wrist
            relation: above
            reference: waist_midpoint
          - subject: left_shoulder
            relation: far
            reference: right_shoulder
            threshold: 0.8
      - name: turned_with_hand_down
        min_frames: 2
        conditions:
          - subject: wrist
            relation: near
            reference: waist_midpoint
            threshold: 0.6
          - subject: left_shoulder
            relation: near
            reference: right_shoulder
            threshold: 0.4
```

### Example F — Arms Raised

**Behavior:** Both arms above head (distress / aggression).

```yaml
arms_raised:
  description: "Both arms raised above head level"
  enabled: false
  alert_type: ZONE_VIOLATION
  pose:
    per_side: false
    phases:
      - name: arms_up
        min_frames: 3
        conditions:
          - subject: left_wrist
            relation: above
            reference: nose
          - subject: right_wrist
            relation: above
            reference: nose
```

### Example G — Fall Detection

**Behavior:** Person falls — head drops to ankle level.

```yaml
fall_detection:
  description: "Person falls to ground — head drops to ankle level"
  enabled: false
  alert_type: ZONE_VIOLATION
  pose:
    per_side: false
    phases:
      - name: upright
        min_frames: 2
        conditions:
          - subject: nose
            relation: above
            reference: waist_midpoint
      - name: on_ground
        min_frames: 2
        conditions:
          - subject: nose
            relation: near
            reference: left_ankle
            threshold: 1.0
```

### Example H — Hand NOT Near Cart (Confirm Concealment)

**Behavior:** Hand is at waist AND not near chest (ruling out open carrying).

```yaml
concealment_confirm:
  description: "Hand at waist and not near chest (not openly carrying)"
  enabled: false
  alert_type: CONCEALMENT
  pose:
    per_side: true
    phases:
      - name: hidden
        min_frames: 3
        conditions:
          - subject: wrist
            relation: near
            reference: waist_midpoint
            threshold: 0.5
          - subject: wrist
            relation: not_near
            reference: chest_midpoint
            threshold: 0.4
```

---

## Reference

### All Keypoints

| Short Name | Expands To (per_side) | Index |
|------------|----------------------|-------|
| `nose` | — (never expands) | 0 |
| `eye` | `left_eye` / `right_eye` | 1, 2 |
| `ear` | `left_ear` / `right_ear` | 3, 4 |
| `shoulder` | `left_shoulder` / `right_shoulder` | 5, 6 |
| `elbow` | `left_elbow` / `right_elbow` | 7, 8 |
| `wrist` | `left_wrist` / `right_wrist` | 9, 10 |
| `hip` | `left_hip` / `right_hip` | 11, 12 |
| `knee` | `left_knee` / `right_knee` | 13, 14 |
| `ankle` | `left_ankle` / `right_ankle` | 15, 16 |

### Virtual Points

| Name | Definition |
|------|-----------|
| `waist_midpoint` | Average of `left_hip` and `right_hip` |
| `chest_midpoint` | Average of `left_shoulder` and `right_shoulder` |
| `torso_center` | Average of `chest_midpoint` and `waist_midpoint` |
| `head_center` | Average of `left_ear` and `right_ear` |

### Relations

| Relation | Condition | Params |
|----------|-----------|--------|
| `above` | subject.y < reference.y | — |
| `below` | subject.y > reference.y | — |
| `left_of` | subject.x < reference.x | — |
| `right_of` | subject.x > reference.x | — |
| `near` | distance < threshold × torso_length | `threshold` |
| `far` | distance ≥ threshold × torso_length | `threshold` |
| `moving_fast` | velocity > threshold × torso_length/frame | `threshold` |
| `stationary` | velocity < threshold × torso_length/frame | `threshold` |
| `bent` | angle at vertex ∈ [min_angle, max_angle] | `reference: [a, c]`, `min_angle`, `max_angle` |
| `straight` | angle at vertex > 150° | `reference: [a, c]` (optional `min_angle`) |
| `not_*` | Negates any of the above | same as negated relation |

### Alert Types

| Type | Description |
|------|-------------|
| `CONCEALMENT` | Item hidden on body / in bag |
| `LOITERING` | Prolonged stationary presence |
| `ZONE_VIOLATION` | Entered restricted area / unusual posture |
| `REPEATED_VISIT` | Same person returns (handled by session manager) |
| `CHECKOUT_BYPASS` | Skipped checkout (handled by zone tracking) |

### Temporal Behavior

| Config | Engine Mode |
|--------|------------|
| Ordered `phases` (no `window_size`) | Sliding split — finds optimal partition |
| `window_size: N` + `phases` | Sliding window — evaluates within N-frame windows |
