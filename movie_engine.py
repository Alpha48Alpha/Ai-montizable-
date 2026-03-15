#!/usr/bin/env python3
"""
Movie Engine — AI-Monetizable Full Animation Production Package Generator
=========================================================================
Generates a complete 10-section movie production package:
  1.  Title
  2.  Runtime
  3.  Characters
  4.  Scene List
  5.  Dialogue
  6.  Subtitles
  7.  Visual Prompts
  8.  Audio Plan
  9.  Assembly Plan
  10. Deliverables

Run:
    python movie_engine.py
"""

import json
import textwrap

# ---------------------------------------------------------------------------
# Sample movie data
# ---------------------------------------------------------------------------

MOVIES = [
    {
        "title": "Echoes of the Forgotten City",
        "tagline": "Some memories are worth fighting to reclaim.",
        "genre": "Anime / Sci-Fi",
        "target_audience": "Teen / Young Adult",
        "total_runtime": "02:30",
        "characters": [
            {
                "name": "KIRA",
                "role": "Protagonist",
                "age": 17,
                "appearance": "Short silver hair, violet eyes, pale skin, wearing a worn leather jacket and cargo pants",
                "personality": "determined, empathetic, impulsive, resourceful",
                "speech_style": "direct and passionate",
                "arc": "Learns to trust others instead of facing every challenge alone",
            },
            {
                "name": "REX",
                "role": "Supporting",
                "age": 19,
                "appearance": "Tall build, dark brown hair, amber eyes, tan skin, wearing a grey hoodie",
                "personality": "calm, analytical, loyal, quietly humorous",
                "speech_style": "measured and thoughtful",
                "arc": "Overcomes his fear of failure to stand by Kira at the climax",
            },
        ],
        "scenes": [
            {
                "number": 1,
                "title": "The Broken Skyline",
                "location": "Exterior — ruined city rooftop",
                "time_of_day": "Dusk",
                "mood": "melancholy, hopeful",
                "start": "00:00",
                "end": "00:45",
                "characters_present": ["KIRA"],
                "dialogue": [
                    ("00:00", "NARRATOR", "In the city where memories go to die, one girl refuses to forget."),
                    ("00:10", "KIRA", "I'll find what they took from us. No matter the cost."),
                ],
                "sfx": ["wind howling", "distant rumble"],
                "music": "ethereal ambient, slow tempo, fade in",
                "visual_prompt": (
                    "Wide establishing shot of a crumbling futuristic cityscape at dusk, "
                    "KIRA standing at the edge of a rooftop, short silver hair blowing in the wind, "
                    "violet eyes gazing at the horizon, worn leather jacket, anime cel-shaded style, "
                    "warm orange and deep purple color palette, cinematic composition, "
                    "high detail, melancholy atmosphere"
                ),
                "transition": "Cross-dissolve",
            },
            {
                "number": 2,
                "title": "Old Allies",
                "location": "Interior — abandoned warehouse",
                "time_of_day": "Night",
                "mood": "tense, reunion",
                "start": "00:45",
                "end": "01:30",
                "characters_present": ["KIRA", "REX"],
                "dialogue": [
                    ("00:45", "REX", "You came back. I wasn't sure you would."),
                    ("00:50", "KIRA", "I need your help, Rex. There's no one else I can trust."),
                    ("00:58", "REX", "Then tell me everything."),
                ],
                "sfx": ["footsteps on concrete", "flickering light buzz"],
                "music": "tense underscore, mid tempo, sustained strings",
                "visual_prompt": (
                    "Medium two-shot inside a dark warehouse lit by a single flickering bulb, "
                    "KIRA facing REX, tension in their postures, "
                    "KIRA: short silver hair, violet eyes, leather jacket; "
                    "REX: tall, dark brown hair, amber eyes, grey hoodie, "
                    "anime style, cool blue and amber contrast lighting, "
                    "cinematic over-the-shoulder angle, high detail"
                ),
                "transition": "Cut",
            },
            {
                "number": 3,
                "title": "The Memory Core",
                "location": "Interior — underground vault",
                "time_of_day": "Unknown (underground)",
                "mood": "epic, triumphant",
                "start": "01:30",
                "end": "02:30",
                "characters_present": ["KIRA", "REX"],
                "dialogue": [
                    ("01:30", "KIRA", "This is it. Everything they erased — it's all here."),
                    ("01:38", "REX", "Kira, once you open that core, there's no going back."),
                    ("01:45", "KIRA", "Good. Because I'm done looking back."),
                    ("02:10", "NARRATOR", "Some memories are worth every scar."),
                ],
                "sfx": ["humming machinery", "energy surge", "triumphant swell"],
                "music": "full orchestral, building to climax, 120 BPM",
                "visual_prompt": (
                    "Wide shot inside a glowing underground vault filled with floating data spheres, "
                    "KIRA reaching toward a pulsing core of light, REX standing behind her in support, "
                    "KIRA: short silver hair lit by blue-white light, violet eyes wide with awe, leather jacket; "
                    "REX: tall silhouette, amber eyes reflecting the glow, "
                    "anime style, electric blue and white color palette, "
                    "dramatic low angle, epic atmosphere, high detail, cinematic composition"
                ),
                "transition": "Fade to black",
            },
        ],
        "export_settings": {
            "resolution": "1920×1080",
            "frame_rate": "24 fps",
            "format": "MP4",
            "codec": "H.264",
            "audio": "AAC 48 kHz stereo",
        },
        "deliverable_label": "concept clip",
    },
]

# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 70
SECTION_SEP = "-" * 70


def section_title(title: str) -> str:
    return f"\n{'─' * 70}\n  {title}\n{'─' * 70}"


def generate_title_section(movie: dict) -> str:
    lines = [
        section_title("1. TITLE"),
        f"  Working Title : {movie['title']}",
        f"  Tagline       : {movie['tagline']}",
        f"  Genre         : {movie['genre']}",
        f"  Audience      : {movie['target_audience']}",
    ]
    return "\n".join(lines)


def generate_runtime_section(movie: dict) -> str:
    lines = [
        section_title("2. RUNTIME"),
        f"  Total Runtime : {movie['total_runtime']}",
        f"  Scenes        : {len(movie['scenes'])}",
        "",
        "  Per-Scene Duration:",
        f"  {'Scene':<8} {'Title':<30} {'Start':<8} {'End':<8} {'Duration':<10}",
        f"  {'─'*8} {'─'*30} {'─'*8} {'─'*8} {'─'*10}",
    ]
    for scene in movie["scenes"]:
        start_sec = _mmss_to_seconds(scene["start"])
        end_sec = _mmss_to_seconds(scene["end"])
        duration = end_sec - start_sec
        lines.append(
            f"  {scene['number']:<8} {scene['title']:<30} {scene['start']:<8} "
            f"{scene['end']:<8} {duration}s"
        )
    return "\n".join(lines)


def generate_characters_section(movie: dict) -> str:
    lines = [section_title("3. CHARACTERS")]
    for char in movie["characters"]:
        lines += [
            "",
            f"  ▸ {char['name']} ({char['role']}, age {char['age']})",
            f"    Appearance : {char['appearance']}",
            f"    Personality: {char['personality']}",
            f"    Speech     : {char['speech_style']}",
            f"    Arc        : {char['arc']}",
        ]
    return "\n".join(lines)


def generate_scene_list_section(movie: dict) -> str:
    lines = [
        section_title("4. SCENE LIST"),
        "",
        f"  {'#':<5} {'Title':<28} {'Location':<30} {'Time':<10} {'Mood':<20} Characters",
        f"  {'─'*5} {'─'*28} {'─'*30} {'─'*10} {'─'*20} {'─'*20}",
    ]
    for scene in movie["scenes"]:
        chars = ", ".join(scene["characters_present"])
        lines.append(
            f"  {scene['number']:<5} {scene['title']:<28} {scene['location'][:28]:<30} "
            f"{scene['time_of_day'][:8]:<10} {scene['mood'][:18]:<20} {chars}"
        )
    return "\n".join(lines)


def generate_dialogue_section(movie: dict) -> str:
    lines = [section_title("5. DIALOGUE")]
    for scene in movie["scenes"]:
        lines += ["", f"  — Scene {scene['number']}: {scene['title']} —"]
        for timecode, speaker, line in scene["dialogue"]:
            lines.append(f"  [{timecode}] {speaker}: {line}")
    return "\n".join(lines)


def generate_subtitles_section(movie: dict) -> str:
    lines = [section_title("6. SUBTITLES"), "", "  (SRT format)"]
    counter = 1
    for scene in movie["scenes"]:
        for timecode, speaker, line in scene["dialogue"]:
            start_s = _mmss_to_seconds(timecode)
            end_s = start_s + 4
            lines += [
                "",
                f"  {counter}",
                f"  {_seconds_to_srt(start_s)} --> {_seconds_to_srt(end_s)}",
                f"  {speaker}: {line}",
            ]
            counter += 1
    return "\n".join(lines)


def generate_visual_prompts_section(movie: dict) -> str:
    lines = [section_title("7. VISUAL PROMPTS")]
    for scene in movie["scenes"]:
        lines += [
            "",
            f"  Scene {scene['number']} — {scene['title']}:",
            "",
        ]
        wrapped = textwrap.fill(scene["visual_prompt"], width=66,
                                initial_indent="    ", subsequent_indent="    ")
        lines.append(wrapped)
    return "\n".join(lines)


def generate_audio_plan_section(movie: dict) -> str:
    lines = [section_title("8. AUDIO PLAN"), "", "  Music:"]
    for scene in movie["scenes"]:
        lines.append(f"    Scene {scene['number']} — {scene['title']}: {scene['music']}")

    lines += ["", "  Sound Effects (SFX):"]
    for scene in movie["scenes"]:
        sfx_str = ", ".join(scene["sfx"])
        lines.append(f"    [{scene['start']}] Scene {scene['number']}: {sfx_str}")

    lines += ["", "  Voice-Over / Narration:"]
    for scene in movie["scenes"]:
        for timecode, speaker, line in scene["dialogue"]:
            if speaker == "NARRATOR":
                wrapped = textwrap.fill(
                    f"[{timecode}] {speaker}: {line}",
                    width=64, initial_indent="    ", subsequent_indent="      "
                )
                lines.append(wrapped)
    return "\n".join(lines)


def generate_assembly_plan_section(movie: dict) -> str:
    lines = [section_title("9. ASSEMBLY PLAN"), "", "  Editing Order:"]
    scenes = movie["scenes"]
    for i, scene in enumerate(scenes):
        slug = scene["title"].lower().replace(" ", "_")
        if i < len(scenes) - 1:
            next_slug = scenes[i + 1]["title"].lower().replace(" ", "_")
            transition = scene["transition"]
            next_num = scenes[i + 1]["number"]
            entry = (f"    {i + 1}. scene_{scene['number']}_{slug}.mp4"
                     f"  → [{transition}] →  scene_{next_num}_{next_slug}.mp4")
            lines.append(entry)
        else:
            lines.append(f"    {i + 1}. scene_{scene['number']}_{slug}.mp4  (final scene)")

    export = movie["export_settings"]
    lines += [
        "",
        "  Export Settings:",
        f"    Resolution : {export['resolution']}",
        f"    Frame Rate : {export['frame_rate']}",
        f"    Format     : {export['format']}",
        f"    Codec      : {export['codec']}",
        f"    Audio      : {export['audio']}",
    ]
    return "\n".join(lines)


def generate_deliverables_section(movie: dict) -> str:
    slug = movie["title"].lower().replace(" ", "_")
    label = movie["deliverable_label"]
    rows = [
        (f"{slug}.mp4", label, "Main video output"),
        (f"{slug}.srt", "subtitles file", "Full subtitle track"),
        (f"{slug}_screenplay.pdf", "screenplay", "Final script"),
        (f"{slug}_storyboard.pdf", "storyboard", "Scene panel layouts"),
        (f"{slug}_prompts.txt", "visual prompts", "All AI generation prompts"),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    lines = [
        section_title("10. DELIVERABLES"),
        "",
        f"  {'File':<{col_w}} {'Quality Label':<22} Description",
        f"  {'─'*col_w} {'─'*22} {'─'*26}",
    ]
    for filename, qlabel, description in rows:
        lines.append(f"  {filename:<{col_w}} {qlabel:<22} {description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Package builder
# ---------------------------------------------------------------------------

def build_movie_package(movie: dict) -> dict:
    return {
        "title": generate_title_section(movie),
        "runtime": generate_runtime_section(movie),
        "characters": generate_characters_section(movie),
        "scene_list": generate_scene_list_section(movie),
        "dialogue": generate_dialogue_section(movie),
        "subtitles": generate_subtitles_section(movie),
        "visual_prompts": generate_visual_prompts_section(movie),
        "audio_plan": generate_audio_plan_section(movie),
        "assembly_plan": generate_assembly_plan_section(movie),
        "deliverables": generate_deliverables_section(movie),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_package(movie: dict) -> None:
    pkg = build_movie_package(movie)

    print(SEPARATOR)
    print(f"  MOVIE PRODUCTION PACKAGE — {movie['title'].upper()}")
    print(SEPARATOR)

    for section in pkg.values():
        print(section)
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mmss_to_seconds(mmss: str) -> int:
    parts = mmss.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _seconds_to_srt(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},000"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n🎬  AI-MONETIZABLE MOVIE ENGINE  🎬")
    print("Generating full 10-section production packages …\n")

    for movie in MOVIES:
        print_package(movie)

    # Export first package as JSON
    sample = MOVIES[0]
    output_path = "movie_engine_output.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"movie": sample["title"], "package": build_movie_package(sample)}, fh, indent=2)

    print(f"✅  Done. Sample package exported to '{output_path}'.")


if __name__ == "__main__":
    main()
