'use strict';

/**
 * Bull job queue — single source of truth for both the API (producer)
 * and the worker process (consumer).
 */

const Queue = require('bull');
const config = require('../shared/config');

const trailerQueue = new Queue(config.queue.name, {
  redis: {
    host: config.redis.host,
    port: config.redis.port,
    password: config.redis.password,
  },
  defaultJobOptions: {
    attempts: 3,
    backoff: { type: 'exponential', delay: 5000 },
    removeOnComplete: config.queue.removeOnComplete,
    removeOnFail: config.queue.removeOnFail,
  },
});

trailerQueue.on('error', (err) => {
  console.error('[queue] Redis error:', err.message);
});

module.exports = trailerQueue;
