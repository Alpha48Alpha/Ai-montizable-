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
├── movie_engine.py                    # Python production package generator
├── train.py                           # RL Lab training entry point
├── templates/
│   ├── movie_production_package.md    # Complete 10-section production template
│   ├── character_sheet.md             # Per-character continuity template
│   └── scene_breakdown.md             # Per-scene storyboard & panel template
├── rl_lab/                            # Production-grade RL Research Lab
│   ├── envs/
│   │   ├── base.py                    # BaseEnv abstract interface
│   │   ├── grid_world.py              # 8×8 discrete navigation task
│   │   └── continuous_world.py        # 2-D continuous navigation task
│   ├── agents/
│   │   ├── base.py                    # BaseAgent abstract interface
│   │   ├── reinforce.py               # REINFORCE (policy gradient + baseline)
│   │   └── dqn.py                     # Double DQN with experience replay
│   ├── models/
│   │   ├── mlp.py                     # MLP, PolicyMLP, ValueMLP
│   │   └── cnn.py                     # CNN encoder (pixel observations)
│   ├── utils/
│   │   ├── replay_buffer.py           # Uniform experience replay buffer
│   │   ├── checkpointing.py           # Save / load .pt checkpoints
│   │   └── metrics.py                 # JSONL metrics logger
│   ├── configs/
│   │   ├── dqn_gridworld.json         # DQN × GridWorld config
│   │   ├── reinforce_gridworld.json   # REINFORCE × GridWorld config
│   │   └── reinforce_continuous.json  # REINFORCE × ContinuousWorld config
│   ├── experiment.py                  # Experiment orchestrator
│   ├── evaluate.py                    # Deterministic evaluation script
│   └── visualize.py                   # Learning-curve & metric plots
└── tests/
    └── test_rl_lab.py                 # 43 pytest tests (all passing)
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

### Movie Engine

1. Submit any movie/video/anime/cartoon/trailer request.
2. The engine generates a full production package using the templates in `/templates/`.
3. Character details from `character_sheet.md` are referenced in every scene to ensure continuity.
4. Each scene uses `scene_breakdown.md` format with visual prompts and timecodes.
5. The final package follows `movie_production_package.md` — all 10 sections complete.

To generate a sample package programmatically:

```bash
python movie_engine.py
```

For detailed behavior rules, see [`movie_engine_config.md`](movie_engine_config.md).

---

## RL Research Lab

A production-grade, modular PyTorch reinforcement-learning framework
that lets agents learn robust goal-directed behavior from trial and error.

### Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib

# Train DQN on GridWorld
python train.py --config rl_lab/configs/dqn_gridworld.json

# Train REINFORCE on GridWorld
python train.py --config rl_lab/configs/reinforce_gridworld.json

# Train REINFORCE on the 2-D continuous navigation task
python train.py --config rl_lab/configs/reinforce_continuous.json

# Evaluate the best saved checkpoint
python -m rl_lab.evaluate --checkpoint runs/dqn_gridworld/checkpoint_best.pt --episodes 20

# Generate learning-curve plots from a completed run
python -m rl_lab.visualize --metrics runs/dqn_gridworld/metrics.jsonl

# Resume training from a checkpoint
python train.py --config rl_lab/configs/dqn_gridworld.json \
                --resume runs/dqn_gridworld/checkpoint_latest.pt

# Run the test suite
python -m pytest tests/test_rl_lab.py -v
```

### Features

| Feature | Details |
|---------|---------|
| **Environments** | GridWorld (discrete 8×8), ContinuousWorld (2-D nav) |
| **Agents** | REINFORCE (policy gradient + value baseline), Double DQN |
| **Neural nets** | MLP, PolicyMLP, ValueMLP, CNN encoder |
| **Replay buffer** | Uniform circular buffer (extensible to PER) |
| **Checkpointing** | Per-episode `.pt` saves + `checkpoint_latest` / `checkpoint_best` symlinks |
| **Metrics logging** | JSONL log with rolling statistics |
| **Evaluation** | Greedy deterministic evaluation with success-rate reporting |
| **Visualization** | Learning curve, eval curve, per-metric plots (PNG) |
| **Configs** | JSON experiment configs for reproducible runs |
| **Tests** | 43 pytest tests covering all modules |

### Extension Points

The codebase is designed to grow into:
- **World models** — attach a latent dynamics model; run imaginary rollouts
- **Multimodal control** — swap the MLP backbone for the `CNNEncoder` to handle pixel observations
- **Human-in-the-loop** — add a `human_step()` method on any env, or attach an RLHF reward model
- **Prioritised replay** — subclass `ReplayBuffer` and override `sample()`
- **PPO / A3C** — extend `REINFORCEAgent` with a rollout buffer and clipped surrogate loss
- **Curriculum learning** — wrap any env in a `CurriculumWrapper` that adjusts difficulty over time

