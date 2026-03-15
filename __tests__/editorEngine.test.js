'use strict';

/**
 * Unit tests for workers/engines/editorEngine.js
 */
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const { assembleTrailer } = require('../workers/engines/editorEngine');

describe('editorEngine', () => {
  let tmpDir;

  const brief = { title: 'Test Movie', genre: 'action' };
  const narrative = {
    hook: 'In the beginning…',
    build: 'Stakes rise…',
    climax: 'The final battle.',
    callToAction: 'Coming soon.',
  };
  const scenes = [
    { startSec: 0, endSec: 10, description: 'Opening', visualPrompt: 'black screen', transition: 'fade' },
    { startSec: 10, endSec: 30, description: 'Action', visualPrompt: 'explosion', transition: 'cut' },
  ];
  const voiceOvers = [
    { beat: 'hook', filePath: '/tmp/vo_hook.mp3' },
  ];
  const music = { trackName: 'epic_action.mp3', bpm: 140, mood: 'intense', durationSec: 60 };

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'editor-test-'));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('writes an EDL JSON file', async () => {
    const result = await assembleTrailer(
      { brief, narrative, scenes, voiceOvers, music },
      tmpDir,
    );
    expect(result).toHaveProperty('edlPath');
    expect(fs.existsSync(result.edlPath)).toBe(true);
  });

  it('EDL contains expected fields', async () => {
    const result = await assembleTrailer(
      { brief, narrative, scenes, voiceOvers, music },
      tmpDir,
    );
    const edl = JSON.parse(fs.readFileSync(result.edlPath, 'utf8'));
    expect(edl.title).toBe(brief.title);
    expect(Array.isArray(edl.videoTracks)).toBe(true);
    expect(Array.isArray(edl.voiceOverTracks)).toBe(true);
    expect(typeof edl.ffmpegHint).toBe('string');
  });

  it('returns a videoPath', async () => {
    const result = await assembleTrailer(
      { brief, narrative, scenes, voiceOvers, music },
      tmpDir,
    );
    expect(result).toHaveProperty('videoPath');
    expect(result.videoPath).toMatch(/\.mp4$/);
  });
});
