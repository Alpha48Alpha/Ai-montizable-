# Ai-montizable- — Full Animation Movie Engine

> **Status: ACTIVE** — All movie, video, anime, cartoon, and trailer requests are
> treated as full production tasks.

---

## What This Engine Enables

With this configuration active, the system automatically creates complete
movie-production packages containing:

| Output | Description |
|--------|-------------|
| 🎬 Movie scripts | Full screenplay with speaker labels and stage directions |
| 🎥 Animation scene breakdowns | Panel-by-panel storyboard with timecodes |
| 🎭 Character sheets | Appearance, personality, and arc details for every character |
| 🖼 Visual generation prompts | Per-scene AI image/video prompts (style, angle, lighting, mood) |
| 🗣 Subtitle scripts | SRT-style subtitle blocks with timestamps |
| 🎧 Audio plans | Music, SFX, and voice-over guidance per scene |
| 🧩 Editing instructions | Assembly order, transitions, and export settings |
| 📦 Full production packages | All of the above in one structured document |

---

## Required Sections in Every Production Package

Every generated package includes **all ten sections**:

1. **Title** — Working title and tagline
2. **Runtime** — Total duration and per-scene time-codes
3. **Characters** — Cast list with consistent appearance and personality details
4. **Scene List** — Ordered scenes with locations and moods
5. **Dialogue** — Complete script with speaker labels
6. **Subtitles** — SRT-format blocks with timestamps
7. **Visual Prompts** — AI generation prompts for every scene
8. **Audio Plan** — Music, SFX, and narration guidance
9. **Assembly Plan** — Editing order, transitions, export settings
10. **Deliverables** — Output file list with honest quality labels

---

## Deliverable Quality Labels

Generated files are labeled with **exactly one** of the following:

| Label | Meaning |
|-------|---------|
| `prototype animation` | Rough motion test; limited frames, placeholder art |
| `concept clip` | Style/mood exploration; not final designs |
| `motion video` | Animatic or motion-graphics with basic movement |
| `slideshow film` | Sequential stills with transitions and audio |
| `rendered short` | Full frame-by-frame rendered animation |

The engine **never** overclaims — a partial motion asset is never described as a
full rendered animation.

---

## Example Prompts That Trigger the Full Pipeline

Users can say things like:

- *"Make me an anime about a girl who discovers magic."*
- *"Create a trailer for my sci-fi movie."*
- *"Write a cartoon episode about a talking robot."*
- *"Generate a short film storyboard."*
- *"Produce a music video concept."*
- *"Build a 2-minute explainer animation."*

All of the above automatically produce the complete 10-section package.

---

## Fallback Pipeline

If direct animation generation is unavailable, the engine **always** produces at
minimum:

1. Screenplay / script
2. Storyboard (panel descriptions)
3. Visual prompts for every scene
4. Subtitle script
5. Scene timing / time-codes
6. Edit / assembly plan

The engine **never** returns an empty or unhelpful response for a production request.

---

## File Structure

```
Ai-montizable-/
├── README.md                          # This file — engine overview
├── movie_engine_config.md             # Full behavior rules & system configuration
└── templates/
    ├── movie_production_package.md    # Complete 10-section production template
    ├── character_sheet.md             # Per-character continuity template
    └── scene_breakdown.md             # Per-scene storyboard & panel template
```

---

## Anime-Specific Behavior

For anime-style requests, the engine additionally:

- Applies consistent anime line-art style tokens to all visual prompts
- Describes panel composition using cinematic terms (close-up, wide shot, Dutch angle, etc.)
- Adds an emotional beat note to every scene
- Maintains a consistent color palette across all character appearances

---

## Usage

1. Submit any movie/video/anime/cartoon/trailer request.
2. The engine generates a full production package using the templates in `/templates/`.
3. Character details from `character_sheet.md` are referenced in every scene to ensure continuity.
4. Each scene uses `scene_breakdown.md` format with visual prompts and timecodes.
5. The final package follows `movie_production_package.md` — all 10 sections complete.

For detailed behavior rules, see [`movie_engine_config.md`](movie_engine_config.md).

