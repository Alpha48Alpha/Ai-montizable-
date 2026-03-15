'use strict';

/**
 * Unit tests for workers/engines/musicEngine.js
 */
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const { selectMusic } = require('../workers/engines/musicEngine');

describe('musicEngine', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'music-test-'));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('returns a music descriptor', async () => {
    const result = await selectMusic({ genre: 'action' }, tmpDir);
    expect(result).toHaveProperty('trackName');
    expect(result).toHaveProperty('bpm');
    expect(result).toHaveProperty('mood');
    expect(result).toHaveProperty('durationSec', 60);
  });

  it('writes a descriptor JSON file', async () => {
    await selectMusic({ genre: 'drama' }, tmpDir);
    const filePath = path.join(tmpDir, 'music_descriptor.json');
    expect(fs.existsSync(filePath)).toBe(true);
    const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    expect(parsed.mood).toBe('emotional');
  });

  it('falls back to default genre for unknown genre', async () => {
    const result = await selectMusic({ genre: 'western' }, tmpDir);
    expect(result.trackName).toBe('cinematic_build.mp3');
  });
});
