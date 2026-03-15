'use strict';

/**
 * Music Engine
 * ─────────────
 * Selects or generates background music for the trailer.
 *
 * Strategy (in priority order):
 *   1. AI-generated track via an external music API (e.g. Suno, Udio).
 *   2. Pick a royalty-free track from a local library based on genre/tone.
 *   3. Stub — write a descriptor JSON so downstream steps can proceed.
 */

const fs   = require('fs');
const fsp  = fs.promises;
const path = require('path');
const config = require('../../shared/config');

/** Map of genre → royalty-free stub track descriptor */
const GENRE_TRACKS = {
  action:   { name: 'epic_action.mp3',   bpm: 140, mood: 'intense'   },
  drama:    { name: 'cinematic_drama.mp3', bpm: 72, mood: 'emotional' },
  horror:   { name: 'dark_tension.mp3',   bpm: 60, mood: 'eerie'     },
  comedy:   { name: 'light_quirky.mp3',   bpm: 120, mood: 'playful'  },
  romance:  { name: 'soft_piano.mp3',     bpm: 80, mood: 'tender'    },
  scifi:    { name: 'synth_space.mp3',    bpm: 110, mood: 'futuristic'},
  default:  { name: 'cinematic_build.mp3', bpm: 90, mood: 'dramatic' },
};

/**
 * @param {object} brief      - original user brief (uses brief.genre / brief.tone)
 * @param {string} outputDir  - directory where music descriptor is written
 * @returns {Promise<object>} music asset descriptor
 */
async function selectMusic(brief, outputDir) {
  console.log('[musicEngine] selecting background music');

  await fsp.mkdir(outputDir, { recursive: true });

  const genre = (brief.genre || 'default').toLowerCase();
  const track = GENRE_TRACKS[genre] || GENRE_TRACKS.default;

  const descriptor = {
    trackName: track.name,
    bpm:       track.bpm,
    mood:      track.mood,
    durationSec: 60,
    fadeInSec:   2,
    fadeOutSec:  3,
    volume:      0.4,         // mix level relative to voice-over
  };

  const descriptorPath = path.join(outputDir, 'music_descriptor.json');
  await fsp.writeFile(descriptorPath, JSON.stringify(descriptor, null, 2));

  return { ...descriptor, descriptorPath };
}

module.exports = { selectMusic };
