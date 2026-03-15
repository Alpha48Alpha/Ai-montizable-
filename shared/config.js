'use strict';

/**
 * Shared configuration used by both the API server and the worker process.
 * Values are read from environment variables with sensible defaults for
 * local / Docker-Compose development.
 */
module.exports = {
  // ── Server ─────────────────────────────────────────────────────────────────
  port: parseInt(process.env.PORT, 10) || 3000,

  // ── Redis / Queue ──────────────────────────────────────────────────────────
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT, 10) || 6379,
    password: process.env.REDIS_PASSWORD || undefined,
  },

  // ── Queue settings ─────────────────────────────────────────────────────────
  queue: {
    name: 'trailer-jobs',
    concurrency: parseInt(process.env.WORKER_CONCURRENCY, 10) || 2,
    /** Maximum number of completed jobs to keep in Redis (not a time value). */
    removeOnComplete: 100,
    /** Maximum number of failed jobs to keep in Redis (not a time value). */
    removeOnFail: 200,
  },

  // ── AI / external service credentials ─────────────────────────────────────
  openai: {
    apiKey: process.env.OPENAI_API_KEY || '',
    model: process.env.OPENAI_MODEL || 'gpt-4o',
  },

  elevenlabs: {
    apiKey: process.env.ELEVENLABS_API_KEY || '',
    voiceId: process.env.ELEVENLABS_VOICE_ID || 'EXAVITQu4vr4xnSDxMaL',
  },

  // ── Output paths (inside containers these map to mounted volumes) ──────────
  output: {
    dir: process.env.OUTPUT_DIR || '/tmp/ai-trailer-studio/output',
  },

  // ── CORS ───────────────────────────────────────────────────────────────────
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
  },
};
