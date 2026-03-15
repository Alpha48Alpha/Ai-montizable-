'use strict';

/**
 * Trailer API Routes
 * ───────────────────
 * POST   /api/trailer/generate      — submit a new trailer job
 * GET    /api/trailer/status/:jobId — poll job progress
 * GET    /api/trailer/download/:jobId — stream the EDL JSON for a completed job
 */

const express      = require('express');
const rateLimit    = require('express-rate-limit');
const { body, param, validationResult } = require('express-validator');
const { enqueueTrailer, getJobStatus, getJobResult } = require('../services/orchestrator');
const fs           = require('fs');

const router = express.Router();

// ── Rate limiters ─────────────────────────────────────────────────────────────
const generateLimiter = rateLimit({
  windowMs: 60 * 1000,       // 1 minute
  max: 10,                   // 10 generate requests per IP per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please try again later.' },
});

const downloadLimiter = rateLimit({
  windowMs: 60 * 1000,       // 1 minute
  max: 30,                   // 30 download requests per IP per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please try again later.' },
});

// ── POST /api/trailer/generate ───────────────────────────────────────────────
router.post(
  '/generate',
  generateLimiter,
  [
    body('title')
      .trim()
      .notEmpty().withMessage('title is required')
      .isLength({ max: 200 }).withMessage('title must be ≤ 200 characters'),
    body('genre')
      .optional()
      .trim()
      .isLength({ max: 50 }),
    body('tone')
      .optional()
      .trim()
      .isLength({ max: 50 }),
    body('keyPoints')
      .optional()
      .isArray({ max: 20 }).withMessage('keyPoints must be an array with at most 20 items'),
    body('keyPoints.*')
      .optional()
      .trim()
      .isString()
      .isLength({ max: 200 }),
  ],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const brief = {
      title:     req.body.title,
      genre:     req.body.genre  || 'drama',
      tone:      req.body.tone   || 'cinematic',
      keyPoints: req.body.keyPoints || [],
    };

    try {
      const { jobId } = await enqueueTrailer(brief);
      return res.status(202).json({
        jobId,
        message: 'Trailer job accepted. Poll /api/trailer/status/:jobId for progress.',
        statusUrl:   `/api/trailer/status/${jobId}`,
        downloadUrl: `/api/trailer/download/${jobId}`,
      });
    } catch (err) {
      console.error('[route] enqueue error:', err);
      return res.status(500).json({ error: 'Failed to enqueue job.' });
    }
  },
);

// ── GET /api/trailer/status/:jobId ───────────────────────────────────────────
router.get(
  '/status/:jobId',
  [param('jobId').trim().notEmpty().isAlphanumeric()],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    try {
      const status = await getJobStatus(req.params.jobId);
      if (!status) {
        return res.status(404).json({ error: 'Job not found.' });
      }
      return res.json(status);
    } catch (err) {
      console.error('[route] status error:', err);
      return res.status(500).json({ error: 'Failed to fetch job status.' });
    }
  },
);

// ── GET /api/trailer/download/:jobId ─────────────────────────────────────────
router.get(
  '/download/:jobId',
  downloadLimiter,
  [param('jobId').trim().notEmpty().isAlphanumeric()],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    try {
      const result = await getJobResult(req.params.jobId);
      if (!result) {
        return res.status(404).json({
          error: 'Job not found or not yet completed.',
        });
      }

      const { edlPath } = result;

      if (!fs.existsSync(edlPath)) {
        return res.status(404).json({ error: 'Output file not found.' });
      }

      res.setHeader('Content-Type', 'application/json');
      res.setHeader(
        'Content-Disposition',
        `attachment; filename="trailer_edl_${req.params.jobId}.json"`,
      );
      fs.createReadStream(edlPath).pipe(res);
    } catch (err) {
      console.error('[route] download error:', err);
      return res.status(500).json({ error: 'Failed to serve download.' });
    }
  },
);

module.exports = router;
