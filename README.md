# 🎬 AI Trailer Studio

An AI-powered platform that generates complete 60-second movie trailers from a simple brief — including story narrative, visual shot list, voice-over audio, background music selection, and a render-ready Edit Decision List (EDL).

---

## Project Structure

```
ai-trailer-studio/
│
├── frontend/
│   ├── public/
│   │   ├── index.html      ← PWA shell
│   │   ├── app.js          ← Frontend logic + polling
│   │   ├── styles.css      ← Dark-mode UI
│   │   └── sw.js           ← Service worker (offline cache)
│   └── package.json
│
├── api/
│   ├── server.js           ← Express HTTP server
│   ├── routes/
│   │   └── trailer.js      ← POST /generate · GET /status/:id · GET /download/:id
│   └── services/
│       └── orchestrator.js ← Enqueue jobs, query status
│
├── workers/
│   ├── worker.js           ← Bull consumer — runs the 5-stage pipeline
│   ├── queue.js            ← Bull queue definition (shared by API + worker)
│   └── engines/
│       ├── storyEngine.js  ← LLM narrative generation
│       ├── sceneEngine.js  ← Shot-list breakdown
│       ├── voiceEngine.js  ← TTS voice-over (ElevenLabs)
│       ├── musicEngine.js  ← Genre-matched background music
│       └── editorEngine.js ← EDL + FFmpeg assembly
│
├── shared/
│   └── config.js           ← Env-driven config (ports, Redis, AI keys)
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.worker
│
├── package.json
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### 1. Docker Compose (recommended)

```bash
# Copy and fill in your AI keys (optional — stubs work without them)
cp .env.example .env   # edit OPENAI_API_KEY and ELEVENLABS_API_KEY

docker-compose up --build
```

| Service  | URL                   |
|----------|-----------------------|
| Frontend | http://localhost:8080 |
| API      | http://localhost:3000 |
| Redis    | localhost:6379        |

### 2. Local development

**Prerequisites:** Node.js ≥ 20, Redis running on `localhost:6379`

```bash
npm install

# Terminal 1 — API server
npm start            # or: npm run dev:api

# Terminal 2 — Worker
npm run worker       # or: npm run dev:worker

# Terminal 3 — Frontend
cd frontend && npm install && npm start
```

---

## API Reference

### `POST /api/trailer/generate`
Submit a new trailer-generation job.

**Request body**
```json
{
  "title":     "Edge of Tomorrow",
  "genre":     "scifi",
  "tone":      "intense",
  "keyPoints": ["soldier trapped in time-loop", "must defeat alien invaders"]
}
```

**Response `202 Accepted`**
```json
{
  "jobId":       "42",
  "statusUrl":   "/api/trailer/status/42",
  "downloadUrl": "/api/trailer/download/42"
}
```

---

### `GET /api/trailer/status/:jobId`
Poll job progress.

**Response**
```json
{
  "jobId":    "42",
  "state":    "active",
  "progress": 55,
  "stage":    "voice"
}
```

States: `waiting` · `active` · `completed` · `failed`  
Stages: `story` → `scenes` → `voice` → `music` → `editor` → `done`

---

### `GET /api/trailer/download/:jobId`
Stream the completed Edit Decision List (EDL JSON) for the job.

---

## Environment Variables

| Variable              | Default                       | Description                        |
|-----------------------|-------------------------------|------------------------------------|
| `PORT`                | `3000`                        | API server port                    |
| `REDIS_HOST`          | `localhost`                   | Redis hostname                     |
| `REDIS_PORT`          | `6379`                        | Redis port                         |
| `REDIS_PASSWORD`      | *(none)*                      | Redis password (if any)            |
| `OPENAI_API_KEY`      | *(none)*                      | GPT-4o for story + scene engines   |
| `OPENAI_MODEL`        | `gpt-4o`                      | OpenAI model name                  |
| `ELEVENLABS_API_KEY`  | *(none)*                      | ElevenLabs TTS for voice engine    |
| `ELEVENLABS_VOICE_ID` | `EXAVITQu4vr4xnSDxMaL`        | ElevenLabs voice ID                |
| `OUTPUT_DIR`          | `/tmp/ai-trailer-studio/output` | Where job outputs are written    |
| `WORKER_CONCURRENCY`  | `2`                           | Parallel jobs per worker process   |
| `CORS_ORIGIN`         | `*`                           | Allowed CORS origin                |

> **Without AI keys** the engines run in stub mode — they return deterministic placeholder content so the full pipeline can be exercised locally.

---

## Five-Stage Pipeline

| # | Engine         | Input              | Output                          |
|---|----------------|--------------------|---------------------------------|
| 1 | Story Engine   | User brief         | Hook / Build / Climax / CTA     |
| 2 | Scene Engine   | Narrative          | Timed shot list + visual prompts|
| 3 | Voice Engine   | Narrative beats    | Per-beat MP3 voice-over files   |
| 4 | Music Engine   | Genre / tone       | Track descriptor + mix settings |
| 5 | Editor Engine  | All of the above   | EDL JSON + FFmpeg render hint   |

---

## Running Tests

```bash
npm test
```

