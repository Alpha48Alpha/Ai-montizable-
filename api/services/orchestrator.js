'use strict';

/**
 * Orchestrator Service
 * ─────────────────────
 * Called by the API route to enqueue a new trailer-generation job and to
 * query job status.  All queue interactions live here so the route handler
 * stays thin.
 */

const trailerQueue = require('../../workers/queue');

/**
 * Enqueue a new trailer job.
 *
 * @param {object} brief  - validated user brief from the request body
 * @returns {Promise<{jobId: string}>}
 */
async function enqueueTrailer(brief) {
  const job = await trailerQueue.add({ brief });
  console.log(`[orchestrator] enqueued job ${job.id} for "${brief.title}"`);
  return { jobId: job.id.toString() };
}

/**
 * Fetch job status and result (if finished).
 *
 * @param {string} jobId
 * @returns {Promise<object>} status object
 */
async function getJobStatus(jobId) {
  const job = await trailerQueue.getJob(jobId);
  if (!job) return null;

  const state    = await job.getState();   // waiting | active | completed | failed | delayed
  const progress = job._progress || 0;
  const stage    = (job.data && job.data._stage) || 'queued';

  const base = {
    jobId,
    state,
    progress,
    stage,
    createdAt: new Date(job.timestamp).toISOString(),
  };

  if (state === 'completed') {
    return { ...base, result: job.returnvalue };
  }

  if (state === 'failed') {
    return { ...base, error: job.failedReason };
  }

  return base;
}

/**
 * Retrieve the output paths for a completed job so the route can
 * stream the file back to the client.
 *
 * @param {string} jobId
 * @returns {Promise<object|null>} { edlPath, videoPath } or null
 */
async function getJobResult(jobId) {
  const job = await trailerQueue.getJob(jobId);
  if (!job) return null;

  const state = await job.getState();
  if (state !== 'completed') return null;

  return job.returnvalue;
}

module.exports = { enqueueTrailer, getJobStatus, getJobResult };
