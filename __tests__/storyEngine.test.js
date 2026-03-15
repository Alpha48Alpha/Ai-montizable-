'use strict';

/**
 * Unit tests for workers/engines/storyEngine.js (stub path — no API key)
 */
const { generateStory } = require('../workers/engines/storyEngine');

describe('storyEngine (stub)', () => {
  it('returns all four narrative beats', async () => {
    const narrative = await generateStory({ title: 'Test Film', genre: 'action', tone: 'intense' });
    expect(narrative).toHaveProperty('hook');
    expect(narrative).toHaveProperty('build');
    expect(narrative).toHaveProperty('climax');
    expect(narrative).toHaveProperty('callToAction');
  });

  it('includes the title in the output', async () => {
    const narrative = await generateStory({ title: 'Unique Title XYZ' });
    const combined = Object.values(narrative).join(' ');
    expect(combined).toMatch(/Unique Title XYZ/);
  });

  it('returns strings for all beats', async () => {
    const narrative = await generateStory({ title: 'Film' });
    for (const [, value] of Object.entries(narrative)) {
      expect(typeof value).toBe('string');
      expect(value.length).toBeGreaterThan(0);
    }
  });
});
