'use strict';

/**
 * API Server
 * ──────────
 * Starts the Express HTTP server that exposes the trailer-generation API.
 */

const express = require('express');
const cors    = require('cors');
const helmet  = require('helmet');
const config  = require('../shared/config');
const trailerRouter = require('./routes/trailer');

const app = express();

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: config.cors.origin }));
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: false }));

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (_req, res) => res.json({ status: 'ok', service: 'ai-trailer-studio-api' }));

// ── Routes ────────────────────────────────────────────────────────────────────
app.use('/api/trailer', trailerRouter);

// ── 404 handler ───────────────────────────────────────────────────────────────
app.use((_req, res) => res.status(404).json({ error: 'Not found.' }));

// ── Global error handler ──────────────────────────────────────────────────────
// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error('[server] unhandled error:', err);
  res.status(500).json({ error: 'Internal server error.' });
});

// ── Start ─────────────────────────────────────────────────────────────────────
if (require.main === module) {
  app.listen(config.port, () => {
    console.log(`[server] listening on port ${config.port}`);
  });
}

module.exports = app;
