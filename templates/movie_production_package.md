# Movie Production Package — Full Template

> **Instructions:** Fill in every section below. Do not omit any section.
> Replace all `[PLACEHOLDER]` values with production-specific content.

---

## 1. Title

| Field | Value |
|-------|-------|
| Working Title | `[TITLE]` |
| Tagline | `[One-sentence hook]` |
| Genre | `[e.g., Anime / Sci-Fi / Comedy / Drama]` |
| Target Audience | `[e.g., Teen / All Ages / Adult]` |

---

## 2. Runtime

| Field | Value |
|-------|-------|
| Total Runtime | `[MM:SS]` |
| Number of Scenes | `[N]` |

**Per-Scene Duration**

| Scene # | Title | Start | End | Duration |
|---------|-------|-------|-----|----------|
| 1 | `[Scene title]` | 00:00 | `[MM:SS]` | `[SS]s` |
| 2 | `[Scene title]` | `[MM:SS]` | `[MM:SS]` | `[SS]s` |
| … | … | … | … | … |

---

## 3. Characters

> Repeat the block below for every character. Consistency across scenes is required.

### Character: `[CHARACTER NAME]`

| Field | Value |
|-------|-------|
| Role | `[Protagonist / Antagonist / Supporting / Narrator]` |
| Age | `[Age or apparent age]` |
| Appearance | `[Hair color, eye color, clothing, distinguishing features]` |
| Personality | `[3–5 adjectives + short description]` |
| Voice/Speech style | `[e.g., calm and measured / energetic and loud]` |
| Character arc | `[Brief description of how the character changes]` |
| Consistency notes | `[Visual or behavioral details that must never change]` |

---

## 4. Scene List

| Scene # | Title | Location | Time of Day | Mood | Characters Present |
|---------|-------|----------|-------------|------|--------------------|
| 1 | `[Title]` | `[Location]` | `[Day/Night/etc.]` | `[Mood]` | `[Names]` |
| 2 | `[Title]` | `[Location]` | `[Day/Night/etc.]` | `[Mood]` | `[Names]` |
| … | … | … | … | … | … |

---

## 5. Dialogue

> Format: `[TIMECODE] CHARACTER: dialogue line`

```
[00:00] NARRATOR: [Opening narration text.]

[00:05] CHARACTER A: [Line of dialogue.]
[00:08] CHARACTER B: [Responding line.]

[00:15] CHARACTER A: [Continued dialogue.]
```

---

## 6. Subtitles

> SRT-style format. One block per line or short phrase.

```srt
1
00:00:00,000 --> 00:00:04,000
[Opening narration text.]

2
00:00:05,000 --> 00:00:08,000
CHARACTER A: [Line of dialogue.]

3
00:00:08,000 --> 00:00:11,000
CHARACTER B: [Responding line.]
```

---

## 7. Visual Prompts

> One prompt per scene, written for AI image or video generation tools.
> Include style keywords, camera angle, lighting, and mood.

### Scene 1 — `[Scene Title]`

```
[Detailed visual prompt. Example:]
Wide establishing shot of [location], [time of day], [art style] style,
[lighting description], [mood/atmosphere], [character A] standing at [position],
[color palette], [camera angle], high detail, cinematic composition.
```

### Scene 2 — `[Scene Title]`

```
[Detailed visual prompt for Scene 2.]
```

*(Add a block for every scene.)*

---

## 8. Audio Plan

### Music

| Scene # | Track / Mood | Tempo | Notes |
|---------|-------------|-------|-------|
| 1 | `[e.g., "ethereal ambient"]` | `[BPM or descriptive]` | `[Loop / fade-in / fade-out]` |
| 2 | `[e.g., "upbeat orchestral"]` | `[BPM or descriptive]` | `[Starts at 00:30]` |

### Sound Effects (SFX)

| Timecode | SFX Description |
|----------|-----------------|
| `[MM:SS]` | `[e.g., footsteps on gravel]` |
| `[MM:SS]` | `[e.g., thunder crack]` |

### Voice-Over / Narration

| Timecode | Speaker | Direction |
|----------|---------|-----------|
| `[MM:SS]` | `[Character / Narrator]` | `[e.g., soft and reflective]` |

---

## 9. Assembly Plan

### Editing Order

1. `[Scene 1 file] → [transition type] → [Scene 2 file]`
2. `[Scene 2 file] → [transition type] → [Scene 3 file]`
3. *(continue for all scenes)*

### Transition Types Used

| Transition | Description | Used Between Scenes |
|------------|-------------|---------------------|
| Cut | Instant switch | `[Scene X → Scene Y]` |
| Fade to black | Gradual dark fade | `[Scene X → Scene Y]` |
| Cross-dissolve | Blend between frames | `[Scene X → Scene Y]` |
| Wipe | Directional slide | `[Scene X → Scene Y]` |

### Export Settings

| Setting | Value |
|---------|-------|
| Resolution | `[e.g., 1920×1080]` |
| Frame Rate | `[e.g., 24 fps]` |
| Format | `[e.g., MP4 / MOV / GIF]` |
| Codec | `[e.g., H.264]` |
| Audio | `[e.g., AAC 48 kHz stereo]` |

---

## 10. Deliverables

> Label every file using the approved quality labels from `movie_engine_config.md`.

| File Name | Quality Label | Description |
|-----------|--------------|-------------|
| `[filename.mp4]` | `[prototype animation / concept clip / motion video / slideshow film / rendered short]` | `[Brief description]` |
| `[filename.srt]` | subtitles file | Full subtitle track |
| `[filename.pdf]` | screenplay | Final script PDF |
| `[storyboard.pdf]` | storyboard | Scene-by-scene panel layouts |
| `[prompts.txt]` | visual prompts | All AI generation prompts |

---

*Template version 1.0 — part of the Full Animation Movie Engine*
