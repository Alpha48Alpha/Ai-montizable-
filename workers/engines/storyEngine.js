'use strict';

/**
 * Story Engine
 * ─────────────
 * Transforms a user brief into a structured trailer narrative:
 *   - Hook  (0–5 s)
 *   - Build (5–25 s)
 *   - Climax (25–40 s)
 *   - Call-to-action (40–60 s)
 *
 * In production this calls an LLM (e.g. OpenAI GPT-4o).
 * The stub below returns deterministic output so the rest of the
 * pipeline can run without live API keys.
 */

const config = require('../../shared/config');

/**
 * @param {object} brief  - { title, genre, tone, keyPoints[] }
 * @returns {Promise<object>} narrative
 */
async function generateStory(brief) {
  console.log('[storyEngine] generating narrative for:', brief.title);

  // ── Production path ───────────────────────────────────────────────────────
  if (config.openai.apiKey) {
    const { OpenAI } = require('openai');
    const client = new OpenAI({ apiKey: config.openai.apiKey });

    const prompt = [
      `You are a professional movie-trailer scriptwriter.`,
      `Create a 60-second trailer script for: "${brief.title}".`,
      `Genre: ${brief.genre || 'drama'}. Tone: ${brief.tone || 'cinematic'}.`,
      `Key points to include: ${(brief.keyPoints || []).join(', ')}.`,
      `Return JSON with keys: hook, build, climax, callToAction (each a string).`,
    ].join('\n');

    const completion = await client.chat.completions.create({
      model: config.openai.model,
      response_format: { type: 'json_object' },
      messages: [{ role: 'user', content: prompt }],
    });

    return JSON.parse(completion.choices[0].message.content);
  }

  // ── Stub path (no API key) ────────────────────────────────────────────────
  const { title = 'Untitled', genre = 'drama', tone = 'cinematic' } = brief;
  return {
    hook: `In a world where everything has changed… "${title}" begins.`,
    build: `A ${tone} ${genre} that challenges everything you thought you knew.`,
    climax: `The moment of truth arrives — and nothing will ever be the same.`,
    callToAction: `"${title}" — coming soon. Don't miss it.`,
  };
}

module.exports = { generateStory };
