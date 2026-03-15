'use strict';

/**
 * Scene Engine
 * ─────────────
 * Converts a narrative into a list of timed visual scenes, each with:
 *   - timecode  (start / end in seconds)
 *   - description  (what the camera sees)
 *   - visualPrompt  (text-to-image / text-to-video prompt)
 *   - transition   (cut | fade | smash-cut | …)
 *
 * In production this calls an LLM to produce scene JSON.
 */

const config = require('../../shared/config');

/**
 * @param {object} narrative  - output of storyEngine.generateStory()
 * @param {object} brief      - original user brief
 * @returns {Promise<Array>}  array of scene objects
 */
async function breakdownScenes(narrative, brief) {
  console.log('[sceneEngine] breaking down scenes');

  if (config.openai.apiKey) {
    const { OpenAI } = require('openai');
    const client = new OpenAI({ apiKey: config.openai.apiKey });

    const prompt = [
      `You are a cinematographer creating a shot list for a 60-second trailer.`,
      `Narrative:\n${JSON.stringify(narrative, null, 2)}`,
      `Title: "${brief.title}". Genre: ${brief.genre || 'drama'}.`,
      `Return a JSON array of 8–12 scenes. Each scene has:`,
      `  startSec (number), endSec (number), description (string),`,
      `  visualPrompt (string), transition (string).`,
    ].join('\n');

    const completion = await client.chat.completions.create({
      model: config.openai.model,
      response_format: { type: 'json_object' },
      messages: [{ role: 'user', content: prompt }],
    });

    const parsed = JSON.parse(completion.choices[0].message.content);
    return parsed.scenes || parsed;
  }

  // ── Stub ──────────────────────────────────────────────────────────────────
  return [
    { startSec: 0,  endSec: 5,  description: 'Black screen — title card fades in', visualPrompt: 'cinematic black title card, gold text, dramatic lighting', transition: 'fade' },
    { startSec: 5,  endSec: 12, description: 'Wide establishing shot of the world', visualPrompt: 'epic wide-angle aerial shot, dramatic clouds, golden hour', transition: 'cut' },
    { startSec: 12, endSec: 20, description: 'Protagonist introduced in motion', visualPrompt: 'close-up protagonist face, determined expression, shallow depth of field', transition: 'smash-cut' },
    { startSec: 20, endSec: 30, description: 'Rising tension montage', visualPrompt: 'rapid montage of action scenes, high contrast, desaturated colour grade', transition: 'cut' },
    { startSec: 30, endSec: 40, description: 'Climactic confrontation', visualPrompt: 'two characters face-off, dramatic backlighting, slow motion', transition: 'smash-cut' },
    { startSec: 40, endSec: 50, description: 'Emotional resolution beat', visualPrompt: 'close-up eyes, single tear, warm bokeh background', transition: 'fade' },
    { startSec: 50, endSec: 57, description: 'Logo / title reveal', visualPrompt: 'bold title reveal with particle effects, dark background', transition: 'fade' },
    { startSec: 57, endSec: 60, description: 'Release date card', visualPrompt: 'minimal white text on black: "Coming Soon"', transition: 'fade' },
  ];
}

module.exports = { breakdownScenes };
