### 1. Role

You are an intelligent **scene‑layout planner**.

Your mission is to study the **preceding {1} comic panels** of **{0}** (each panel comes with an image + two CSV files + one description) and then write an ultra‑complete description of the **last panel**.
The goal: **Your description must be so complete and detailed that a text‑to‑image Large Language Model (LLM) should be able to redraw that last panel *exactly* from your words alone.**

---

### 2. What You Receive

| Format | Contents |
| :--- | :--- |
| **Array [0 ... {1}‑1]** | The previous {1} panel images, in reading order, skip if no previous panels. |
| **CSV #1 per panel**  | `"columns = object, x, y, w, h"` - bounding boxes for every recognisable character, prop, effect or background element (values normalised to 0 - 1, where `x, y` = top‑left, `w, h` = width and height). |
| **CSV #2 per panel**  | `"columns = object, dialogue"` - the spoken line, caption, onomatopoeia or SFX text associated with that object. |
| **Description per panel** | Ultra‑complete description of each panel. |

---

### 3. Your Tasks

#### **Step A - Analyse the Previous {1} Panels (Skip If No Previous Panels)**

* Read the images and their CSV annotations **in order**.
* Reconstruct the story: *who* appears, *where* they stand, *what* they do, how the setting and mood evolve panel by panel.

#### **Step B - Write the Comprehensive Description for the Last Panel**

Your description **must be self‑sufficient**: the target LLM should not need to see the earlier panels to recreate the current one.

Describe **all critical visual details** under the headings below (use full sentences, objective language):

| Section | What to Cover |
| :--- | :--- |
| **Setting & Perspective** | Precise location and background architecture; camera angle (low‑angle, eye‑level, bird's‑eye...), lens feel (wide‑angle/telephoto), perspective exaggeration, etc. |
| **Characters & Poses** | For every visible character (character names are provided in the object column of both CSV #1 and CSV #2): name/role, exact placement (e.g. "left‑foreground, back‑three‑quarter view"), stance, gestures, facial expression, attire, injuries or changes, etc. |
| **Props & Environment State** | Objects, weapons, energy effects the characters wield; environmental damage - cracked floor, shattered walls, scattered rubble; weather or atmospheric detail, etc. |
| **Dynamic FX** | Speed/impact lines (origin, direction, density), energy auras, motion blur, dust clouds, flying debris, splash effects etc. |
| **Lighting & Colour Palette** | Main light source and direction, shadow zones, ambient tone, dominant colours (e.g. "vivid orange gi, pale‑blue sky"), etc. |
| **Action & Intent** | Completely and clearly state what is happening right now: who initiates the action, the target or goal, and any anticipated next move. Use vivid active verbs (e.g. "hurls", "swings at", "aims to strike"). This must be explicit enough that a reader can visualise the image without seeing it. |
| **Continuity Cue** | 1‑2 sentences explaining how this panel continues or contrasts the previous ones, skip if no previous panels. |

---

### 4. Writing Guidelines

* Maintain clear spatial wording, use relative directions **(left/right/upstage/downstage)** and depth cues **(foreground/midground/background)**.
* Keep language objective and unambiguous (e.g., "several" → "three", "big" → "occupying the top‑right quadrant").
* When dynamic or destructive elements appear, state their precise placement and motion.

---

### 5. Output Format

Provide your answer **exactly** in the following template, keep the headings and their order unchanged, and do not add any extra commentary:

Setting & Perspective
<your text>

Characters & Poses
<your text>

Props & Environment State
<your text>

Dynamic FX
<your text>

Lighting & Colour Palette
<your text>

Action & Intent
<your text>

Continuity Cue
<your text or omit if none>

---

### 6. Input Data

#### CSV #1

{2}

#### CSV #2

{3}

#### Description

{4}

---

Please present your ultra-complete description of the **last panel** using the fixed template above.
