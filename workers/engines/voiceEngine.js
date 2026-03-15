'use strict';

/**
 * Voice Engine
 * ─────────────
 * Generates voice-over audio for each narrative beat using a TTS service
 * (ElevenLabs in production).  Returns an array of audio asset descriptors
 * with a local file path or base64 payload.
 */

const fs   = require('fs');
const fsp  = fs.promises;
const path = require('path');
const config = require('../../shared/config');

/**
 * @param {object} narrative  - output of storyEngine.generateStory()
 * @param {string} outputDir  - directory where audio files are written
 * @returns {Promise<Array>}  array of { beat, filePath } objects
 */
async function generateVoiceOver(narrative, outputDir) {
  console.log('[voiceEngine] generating voice-over');

  await fsp.mkdir(outputDir, { recursive: true });

  const beats = ['hook', 'build', 'climax', 'callToAction'];
  const results = [];

  for (const beat of beats) {
    const text = narrative[beat];
    if (!text) continue;

    const filePath = path.join(outputDir, `vo_${beat}.mp3`);

    if (config.elevenlabs.apiKey) {
      // ── Production: call ElevenLabs TTS ─────────────────────────────────
      const https = require('https');
      const body  = JSON.stringify({
        text,
        model_id: 'eleven_monolingual_v1',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      });

      const audioBuffer = await new Promise((resolve, reject) => {
        const req = https.request(
          {
            hostname: 'api.elevenlabs.io',
            path: `/v1/text-to-speech/${config.elevenlabs.voiceId}`,
            method: 'POST',
            headers: {
              'xi-api-key': config.elevenlabs.apiKey,
              'Content-Type': 'application/json',
              'Content-Length': Buffer.byteLength(body),
            },
          },
          (res) => {
            const chunks = [];
            res.on('data', (c) => chunks.push(c));
            res.on('end', () => {
              if (res.statusCode !== 200) {
                return reject(new Error(`ElevenLabs error ${res.statusCode}`));
              }
              resolve(Buffer.concat(chunks));
            });
          },
        );
        req.on('error', reject);
        req.write(body);
        req.end();
      });
      await fsp.writeFile(filePath, audioBuffer);
    } else {
      // ── Stub: write a placeholder text file ────────────────────────────
      await fsp.writeFile(filePath, `[STUB VO - ${beat}]: ${text}`);
    }

    results.push({ beat, filePath });
  }

  return results;
}

module.exports = { generateVoiceOver };
