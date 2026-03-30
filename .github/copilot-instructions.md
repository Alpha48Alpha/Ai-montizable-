# GitHub Copilot Instructions — Alpha48Alpha AI Lab

## Project Overview
This repository contains the **Alpha48Alpha AI Lab** — a Python reinforcement learning
research environment. It also houses the **Ai-montizable Animation Engine** for movie
production packages.

## AI Lab Guidelines

- All RL agent code lives under `alpha48alpha_ai_lab/agents/`.
- Environments implement `reset()`, `step(action)`, `render()`, and `get_state()`.
- All models are PyTorch `nn.Module` subclasses located in `alpha48alpha_ai_lab/models/`.
- Training logic belongs in `alpha48alpha_ai_lab/training/trainer.py`.
- Use `alpha48alpha_ai_lab/utils/logger.py` for all metric logging.
- Configuration constants go in `alpha48alpha_ai_lab/config.py`.

## Code Style
- Python 3.10+, typed hints encouraged.
- Detailed docstrings on every class and public method.
- Keep each file focused on a single responsibility.

## Animation Engine Guidelines
- Every production package must include all 10 required sections
  (Title, Runtime, Characters, Scene List, Dialogue, Subtitles,
  Visual Prompts, Audio Plan, Assembly Plan, Deliverables).
- Use `templates/movie_production_package.md` as the canonical template.
- Never overclaim quality — always apply the correct deliverable quality label.
