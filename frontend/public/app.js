/* global state */
// API_BASE can be overridden by setting window.API_BASE before loading this script.
// When the frontend and API are on different origins (e.g. in docker-compose where
// frontend runs on :8080 and API on :3000), set:
//   <script>window.API_BASE = 'http://localhost:3000/api';</script>
// before loading app.js, or configure a reverse proxy.
const API_BASE = window.API_BASE || '/api';

let currentJobId  = null;
let pollInterval  = null;

const STAGES_ORDER = ['story', 'scenes', 'voice', 'music', 'editor', 'done'];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const form            = document.getElementById('trailerForm');
const submitBtn       = document.getElementById('submitBtn');
const progressSection = document.getElementById('progressSection');
const progressBar     = document.getElementById('progressBar');
const progressLabel   = document.getElementById('progressLabel');
const resultSection   = document.getElementById('resultSection');
const downloadBtn     = document.getElementById('downloadBtn');
const errorSection    = document.getElementById('errorSection');
const errorMessage    = document.getElementById('errorMessage');
const narrativePreview = document.getElementById('narrativePreview');

// ── Form submit ───────────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearFieldErrors();
  hideAll();

  const title     = document.getElementById('title').value.trim();
  const genre     = document.getElementById('genre').value;
  const tone      = document.getElementById('tone').value;
  const rawPoints = document.getElementById('keyPoints').value;

  // Client-side validation
  if (!title) {
    showFieldError('titleError', 'Title is required.');
    return;
  }

  const keyPoints = rawPoints
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .slice(0, 20);

  submitBtn.disabled = true;
  submitBtn.textContent = 'Submitting…';

  try {
    const res  = await fetch(`${API_BASE}/trailer/generate`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ title, genre, tone, keyPoints }),
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = (data.errors && data.errors[0]?.msg) || data.error || 'Submission failed.';
      showError(msg);
      return;
    }

    currentJobId = data.jobId;
    showProgress();
    startPolling(data.jobId);
  } catch (err) {
    showError('Network error — is the API server running?');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Generate Trailer';
  }
});

// ── Polling ───────────────────────────────────────────────────────────────────
function startPolling(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(() => pollStatus(jobId), 2000);
}

async function pollStatus(jobId) {
  try {
    const res  = await fetch(`${API_BASE}/trailer/status/${jobId}`);
    const data = await res.json();

    if (!res.ok) {
      stopPolling();
      showError(data.error || 'Failed to fetch status.');
      return;
    }

    updateProgressUI(data);

    if (data.state === 'completed') {
      stopPolling();
      showResult(data.result, jobId);
    } else if (data.state === 'failed') {
      stopPolling();
      showError(data.error || 'Job failed. Please try again.');
    }
  } catch (_err) {
    // silently retry; transient network hiccup
  }
}

function stopPolling() {
  clearInterval(pollInterval);
  pollInterval = null;
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function showProgress() {
  progressSection.classList.remove('hidden');
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Queued…';
  resetStages();
}

function updateProgressUI(data) {
  const pct   = typeof data.progress === 'number' ? data.progress : 0;
  const stage = data.stage || 'queued';

  progressBar.style.width = `${pct}%`;
  progressLabel.textContent = `${stage.charAt(0).toUpperCase() + stage.slice(1)} — ${pct}%`;

  // Highlight stages
  STAGES_ORDER.forEach((s) => {
    const el = document.querySelector(`.stage[data-stage="${s}"]`);
    if (!el) return;
    const stageIdx   = STAGES_ORDER.indexOf(s);
    const currentIdx = STAGES_ORDER.indexOf(stage);
    if (stageIdx < currentIdx) {
      el.classList.remove('active');
      el.classList.add('done');
    } else if (stageIdx === currentIdx) {
      el.classList.add('active');
      el.classList.remove('done');
    } else {
      el.classList.remove('active', 'done');
    }
  });
}

function resetStages() {
  document.querySelectorAll('.stage').forEach((el) => {
    el.classList.remove('active', 'done');
  });
}

function showResult(result, jobId) {
  // Mark all stages done
  document.querySelectorAll('.stage').forEach((el) => {
    el.classList.remove('active');
    el.classList.add('done');
  });
  progressBar.style.width = '100%';
  progressLabel.textContent = 'Complete!';

  resultSection.classList.remove('hidden');

  if (result && result.narrative) {
    const n = result.narrative;
    narrativePreview.innerHTML = [
      n.hook        ? `<p><strong>Hook:</strong> ${escHtml(n.hook)}</p>` : '',
      n.build       ? `<p><strong>Build:</strong> ${escHtml(n.build)}</p>` : '',
      n.climax      ? `<p><strong>Climax:</strong> ${escHtml(n.climax)}</p>` : '',
      n.callToAction? `<p><strong>CTA:</strong> ${escHtml(n.callToAction)}</p>` : '',
    ].join('');
  }

  downloadBtn.href = `${API_BASE}/trailer/download/${jobId}`;
  downloadBtn.setAttribute('download', `trailer_edl_${jobId}.json`);
}

function showError(msg) {
  errorSection.classList.remove('hidden');
  errorMessage.textContent = msg;
}

function hideAll() {
  progressSection.classList.add('hidden');
  resultSection.classList.add('hidden');
  errorSection.classList.add('hidden');
}

function clearFieldErrors() {
  document.querySelectorAll('.field-error').forEach((el) => (el.textContent = ''));
}

function showFieldError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Service Worker registration ───────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/sw.js')
      .then(() => console.log('[app] service worker registered'))
      .catch((err) => console.warn('[app] service worker error:', err));
  });
}
