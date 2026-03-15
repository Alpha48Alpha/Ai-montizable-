'use strict';

/**
 * Unit tests for workers/engines/sceneEngine.js (stub path — no API key)
 */
const { breakdownScenes } = require('../workers/engines/sceneEngine');

describe('sceneEngine (stub)', () => {
  const narrative = {
    hook: 'It begins…',
    build: 'Rising stakes…',
    climax: 'The confrontation.',
    callToAction: 'Coming soon.',
  };

  it('returns an array of scenes', async () => {
    const scenes = await breakdownScenes(narrative, { title: 'Test', genre: 'drama' });
    expect(Array.isArray(scenes)).toBe(true);
    expect(scenes.length).toBeGreaterThanOrEqual(1);
  });

  it('each scene has required fields', async () => {
    const scenes = await breakdownScenes(narrative, { title: 'Test' });
    for (const scene of scenes) {
      expect(scene).toHaveProperty('startSec');
      expect(scene).toHaveProperty('endSec');
      expect(scene).toHaveProperty('description');
      expect(scene).toHaveProperty('visualPrompt');
      expect(scene).toHaveProperty('transition');
    }
  });

  it('scenes have non-negative timecodes', async () => {
    const scenes = await breakdownScenes(narrative, { title: 'Test' });
    for (const scene of scenes) {
      expect(scene.startSec).toBeGreaterThanOrEqual(0);
      expect(scene.endSec).toBeGreaterThan(scene.startSec);
    }
  });
});
