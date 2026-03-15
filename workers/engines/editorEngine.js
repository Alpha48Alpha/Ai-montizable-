'use strict';

/**
 * Editor Engine
 * ─────────────
 * Assembles the final trailer from:
 *   - scenes (timed visual assets)
 *   - voiceOvers (per-beat audio files)
 *   - music (background track descriptor)
 *
 * In production this generates an FFmpeg command (or calls a cloud
 * video-editing API) and produces the final MP4.
 *
 * The stub writes a detailed edit-decision-list (EDL) JSON so the output
 * can be reviewed / handed off to a human editor.
 */

const fs   = require('fs');
const fsp  = fs.promises;
const path = require('path');

/**
 * @param {object} job
 * @param {object} job.brief
 * @param {object} job.narrative
 * @param {Array}  job.scenes
 * @param {Array}  job.voiceOvers
 * @param {object} job.music
 * @param {string} outputDir
 * @returns {Promise<object>} result descriptor { edlPath, videoPath? }
 */
async function assembleTrailer({ brief, narrative, scenes, voiceOvers, music }, outputDir) {
  console.log('[editorEngine] assembling trailer');

  await fsp.mkdir(outputDir, { recursive: true });

  // Build an Edit Decision List (EDL)
  const edl = {
    title: brief.title,
    totalDurationSec: 60,
    createdAt: new Date().toISOString(),
    soundtrack: music,
    voiceOverTracks: voiceOvers.map((vo) => ({
      beat: vo.beat,
      source: vo.filePath,
      text: narrative[vo.beat],
    })),
    videoTracks: scenes.map((scene) => ({
      startSec:     scene.startSec,
      endSec:       scene.endSec,
      description:  scene.description,
      visualPrompt: scene.visualPrompt,
      transition:   scene.transition,
    })),
    ffmpegHint: _buildFfmpegHint(scenes, voiceOvers, music, outputDir),
  };

  const edlPath = path.join(outputDir, 'trailer_edl.json');
  await fsp.writeFile(edlPath, JSON.stringify(edl, null, 2));

  // In a real implementation, shell out to FFmpeg here and set videoPath.
  const videoPath = path.join(outputDir, 'trailer_preview.mp4');

  return { edlPath, videoPath, edl };
}

/**
 * Generates a representative FFmpeg filter-graph hint for documentation.
 * Not executed directly — shows the editor what the final command would look like.
 */
function _buildFfmpegHint(scenes, voiceOvers, music, outputDir) {
  const voInputs  = voiceOvers.map((vo, i) => `-i "${vo.filePath}"`).join(' ');
  const musicHint = music.trackName ? `-i "music/${music.trackName}"` : '';
  const output    = path.join(outputDir, 'trailer_preview.mp4');

  return [
    `ffmpeg`,
    `-f lavfi -i "color=c=black:s=1920x1080:d=60"`,
    voInputs,
    musicHint,
    `-filter_complex "[0:v]scale=1920:1080[v]"`,
    `-map "[v]"`,
    `-c:v libx264 -preset fast -crf 18`,
    `-c:a aac -b:a 192k`,
    `-t 60`,
    `"${output}"`,
  ].filter(Boolean).join(' \\\n  ');
}

module.exports = { assembleTrailer };
