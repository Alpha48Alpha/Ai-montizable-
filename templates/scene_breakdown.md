# Scene Breakdown & Storyboard Template

> **Purpose:** Provide a panel-by-panel breakdown for every scene, combining
> storyboard direction, visual prompts, dialogue cues, and timing.
> Complete one file per scene (or combine all scenes in one file for short films).

---

## Scene Header

| Field | Value |
|-------|-------|
| Scene Number | `[#]` |
| Scene Title | `[Descriptive title]` |
| Location | `[Interior / Exterior — specific place]` |
| Time of Day | `[Dawn / Morning / Noon / Dusk / Night]` |
| Mood / Tone | `[e.g., tense, playful, melancholy, epic]` |
| Start Timecode | `[MM:SS]` |
| End Timecode | `[MM:SS]` |
| Scene Duration | `[SS]s` |
| Characters Present | `[Comma-separated names]` |

---

## Scene Summary

> One paragraph describing what happens in this scene and why it matters to the story.

`[Write summary here.]`

---

## Panel Breakdown

> Repeat the panel block below for every shot in the scene.
> Panels are numbered sequentially across the **entire film** (not reset per scene).
> Example: if Scene 1 ends on Panel 4, Scene 2 begins with Panel 5.

---

### Panel `[N]` — `[Shot name / description]`

| Field | Value |
|-------|-------|
| Panel Timecode | `[MM:SS:FF]` — `[MM:SS:FF]` |
| Duration | `[N]` frames / `[N.N]`s |
| Shot Type | `[Extreme close-up / Close-up / Medium / Wide / Extreme wide / Bird's eye / Worm's eye / Over-the-shoulder / POV]` |
| Camera Movement | `[Static / Pan left / Pan right / Tilt up / Tilt down / Zoom in / Zoom out / Dolly / Tracking shot]` |
| Angle | `[Eye-level / Low angle / High angle / Dutch angle]` |
| Characters | `[Names and positions in frame]` |
| Action | `[What is happening in this frame/shot?]` |
| Emotion Beat | `[e.g., fear, joy, determination, sorrow]` |

**Visual Prompt (AI generation):**

```
[Shot type], [angle], [location] background, [character descriptions],
[action/pose], [lighting: e.g., golden hour / harsh neon / soft diffuse],
[art style: e.g., anime cel-shaded / realistic / cartoon], [color palette],
[mood/atmosphere keywords], [additional style tokens]
```

**Dialogue / SFX at this panel:**

```
[MM:SS] CHARACTER: [Line of dialogue, if any]
[MM:SS] SFX: [Sound effect description, if any]
```

---

*(Copy the Panel block above for each additional panel in this scene.)*

---

## Scene Transition

| Field | Value |
|-------|-------|
| Transition type | `[Cut / Fade to black / Cross-dissolve / Wipe / Smash cut / Iris]` |
| Transition duration | `[N]` frames / `[N.N]`s |
| Leads to Scene # | `[#]` |
| Transition note | `[Optional: what emotional/narrative shift does this signal?]` |

---

## Scene Checklist

Before finalizing this scene, verify:

- [ ] All characters' appearances match their character sheets
- [ ] Every panel has a visual prompt
- [ ] Timecodes are consistent with the master runtime in the production package
- [ ] All dialogue lines appear in the subtitle block
- [ ] Audio plan references this scene's music/SFX
- [ ] Transition to the next scene is defined

---

*Template version 1.0 — part of the Full Animation Movie Engine*
