'use strict';

/**
 * Worker Process
 * ───────────────
 * Pulls trailer-generation jobs from the Bull queue and runs each job
 * through the five-stage pipeline:
 *
 *   1. Story   → narrative beats
 *   2. Scene   → timed visual shot list
 *   3. Voice   → per-beat audio files
 *   4. Music   → background track descriptor
 *   5. Editor  → assembled trailer + EDL
 */

const path = require('path');
const config = require('../shared/config');
const trailerQueue = require('./queue');

const { generateStory }   = require('./engines/storyEngine');
const { breakdownScenes } = require('./engines/sceneEngine');
const { generateVoiceOver } = require('./engines/voiceEngine');
const { selectMusic }      = require('./engines/musicEngine');
const { assembleTrailer }  = require('./engines/editorEngine');

trailerQueue.process(config.queue.concurrency, async (job) => {
  const { brief } = job.data;
  const jobOutputDir = path.join(config.output.dir, job.id.toString());

  console.log(`[worker] starting job ${job.id} — "${brief.title}"`);

  try {
    // Stage 1 — Story
    await _updateProgress(job, 'story', 10);
    const narrative = await generateStory(brief);

    // Stage 2 — Scenes
    await _updateProgress(job, 'scenes', 30);
    const scenes = await breakdownScenes(narrative, brief);

    // Stage 3 — Voice-over
    await _updateProgress(job, 'voice', 55);
    const voiceOvers = await generateVoiceOver(narrative, path.join(jobOutputDir, 'audio'));

    // Stage 4 — Music
    await _updateProgress(job, 'music', 70);
    const music = await selectMusic(brief, path.join(jobOutputDir, 'music'));

    // Stage 5 — Editor
    await _updateProgress(job, 'editor', 90);
    const result = await assembleTrailer(
      { brief, narrative, scenes, voiceOvers, music },
      path.join(jobOutputDir, 'video'),
    );

    await _updateProgress(job, 'done', 100);

    console.log(`[worker] job ${job.id} complete — EDL: ${result.edlPath}`);

    return {
      jobId:     job.id,
      edlPath:   result.edlPath,
      videoPath: result.videoPath,
      narrative,
      scenes,
    };
  } catch (err) {
    console.error(`[worker] job ${job.id} failed:`, err);
    throw err;
  }
});

trailerQueue.on('completed', (job, result) => {
  console.log(`[worker] ✅ job ${job.id} completed`);
});

trailerQueue.on('failed', (job, err) => {
  console.error(`[worker] ❌ job ${job.id} failed: ${err.message}`);
});

console.log(
  `[worker] ready — queue "${config.queue.name}", concurrency ${config.queue.concurrency}`,
);

// ── Helpers ──────────────────────────────────────────────────────────────────

async function _updateProgress(job, stage, percent) {
  await job.progress(percent);
  await job.update({ ...job.data, _stage: stage });
}
