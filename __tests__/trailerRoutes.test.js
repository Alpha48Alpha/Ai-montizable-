'use strict';

/**
 * Integration tests for the Trailer API routes.
 * Uses supertest to exercise the Express app without a live server.
 * Mocks Bull queue so Redis is not required.
 */

jest.mock('../workers/queue', () => {
  const EventEmitter = require('events');
  class MockQueue extends EventEmitter {
    async add(data) {
      return { id: '99', data };
    }
    async getJob(id) {
      if (id === '99') {
        return {
          id: '99',
          data: { brief: { title: 'Mock Film' } },
          _progress: 50,
          timestamp: Date.now(),
          getState: async () => 'active',
          returnvalue: null,
        };
      }
      if (id === 'done') {
        return {
          id: 'done',
          data: { brief: { title: 'Done Film' }, _stage: 'done' },
          _progress: 100,
          timestamp: Date.now(),
          getState: async () => 'completed',
          returnvalue: { edlPath: '/tmp/test_edl.json', videoPath: '/tmp/test.mp4', narrative: {} },
        };
      }
      return null;
    }
    on() { return this; }
  }
  return new MockQueue();
});

const request = require('supertest');
const app     = require('../api/server');

describe('GET /health', () => {
  it('returns 200 ok', async () => {
    const res = await request(app).get('/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
  });
});

describe('POST /api/trailer/generate', () => {
  it('returns 202 with jobId for valid brief', async () => {
    const res = await request(app)
      .post('/api/trailer/generate')
      .send({ title: 'My Movie', genre: 'action', tone: 'intense' });
    expect(res.status).toBe(202);
    expect(res.body).toHaveProperty('jobId');
    expect(res.body).toHaveProperty('statusUrl');
    expect(res.body).toHaveProperty('downloadUrl');
  });

  it('returns 400 when title is missing', async () => {
    const res = await request(app)
      .post('/api/trailer/generate')
      .send({ genre: 'action' });
    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty('errors');
  });

  it('returns 400 when title exceeds 200 chars', async () => {
    const res = await request(app)
      .post('/api/trailer/generate')
      .send({ title: 'A'.repeat(201) });
    expect(res.status).toBe(400);
  });
});

describe('GET /api/trailer/status/:jobId', () => {
  it('returns status for an active job', async () => {
    const res = await request(app).get('/api/trailer/status/99');
    expect(res.status).toBe(200);
    expect(res.body.jobId).toBe('99');
    expect(res.body.state).toBe('active');
  });

  it('returns 404 for unknown jobId', async () => {
    const res = await request(app).get('/api/trailer/status/unknown1');
    expect(res.status).toBe(404);
  });
});

describe('GET /api/trailer/download/:jobId', () => {
  it('returns 404 for unknown jobId', async () => {
    const res = await request(app).get('/api/trailer/download/unknown2');
    expect(res.status).toBe(404);
  });

  it('returns 404 for active (not completed) job', async () => {
    const res = await request(app).get('/api/trailer/download/99');
    expect(res.status).toBe(404);
  });
});

describe('GET /unknown-route', () => {
  it('returns 404', async () => {
    const res = await request(app).get('/not-a-route');
    expect(res.status).toBe(404);
  });
});
