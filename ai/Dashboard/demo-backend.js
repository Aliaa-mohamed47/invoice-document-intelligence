/* demo-backend.js
 * Fake backend for the dashboard. Intercepts fetch() calls to the AWS URL
 * so index.html works with NO AWS. The original AWS code is untouched.
 *
 * MODE: 'demo' = always fake | 'aws' = always real | 'auto' = try AWS, fall back to demo
 */
(function () {
  const MODE = 'demo';
  const AWS_URL = 'http://127.0.0.1:8000';
  const DB_KEY = 'ii-demo-db-v1';

  const INFERENCE_URL = 'http://127.0.0.1:8000';
  
  const REAL_RESULTS = {
    'X51005447850': {
      company: 'REPLACE WITH REAL COMPANY', date: '00/00/0000', total: '0.00', address: 'REPLACE WITH REAL ADDRESS',
      scores: { company: 0.97, date: 0.95, total: 0.99, address: 0.88 }
    },
    'X51005568889': {
      company: 'REPLACE WITH REAL COMPANY', date: '00/00/0000', total: '0.00', address: 'REPLACE WITH REAL ADDRESS',
      scores: { company: 0.96, date: 0.93, total: 0.98, address: 0.85 }
    },
    'X51006008089': {
      company: 'REPLACE WITH REAL COMPANY', date: '00/00/0000', total: '0.00', address: 'REPLACE WITH REAL ADDRESS',
      scores: { company: 0.94, date: 0.97, total: 0.99, address: 0.81 }
    }
  };

  let demoActive = MODE === 'demo';
  const load = () => { try { return JSON.parse(localStorage.getItem(DB_KEY)); } catch { return null; } };
  const save = (db) => { try { localStorage.setItem(DB_KEY, JSON.stringify(db)); } catch {} };

  function makeInvoice(n, filename, r, status) {
    return {
      id: 'demo-' + Date.now() + '-' + n,
      invoice_id: 'INV_' + String(n).padStart(4, '0'),
      filename,
      company: r.company, date: r.date, total: r.total, address: r.address,
      company_score: r.scores.company, date_score: r.scores.date,
      total_score: r.scores.total, address_score: r.scores.address,
      status: status || 'PENDING',
      processing_time: 900 + Math.round(Math.random() * 900),
      created_at: new Date().toISOString()
    };
  }

  function getDB() {
    let db = load();
    if (!db) {
      db = { counter: 0, invoices: [] };
      Object.entries(REAL_RESULTS).forEach(([k, r], i) => {
        db.counter++;
        const inv = makeInvoice(db.counter, k + '.jpg', r, i === 0 ? 'CONFIRMED' : 'PENDING');
        inv.pipeline_doc_id = inv.id;
        db.invoices.push(inv);
      });
      save(db);
    }
    return db;
  }

  const json = (data, status = 200) =>
    new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });

  const stem = (name) => name.replace(/\.[^.]+$/, '');
  const hash = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);

  async function handle(path, opts) {
    const method = (opts.method || 'GET').toUpperCase();
    const db = getDB();
    const url = new URL(path, 'http://x');
    const p = url.pathname;

    if (p === '/health') {
      if (!INFERENCE_URL) return json({ status: 'ok', mode: 'demo' });
      try {
        const h = await realFetch(INFERENCE_URL + '/health');
        if (!h.ok) throw new Error('unhealthy');
        return json({ status: 'ok', mode: 'demo+model' });
      } catch {
        return json({ status: 'model offline' }, 503);
      }
    }

    if (p === '/api/invoices' && method === 'GET')
      return json([...db.invoices].sort((a, b) => b.created_at.localeCompare(a.created_at)));

    if (p === '/api/invoices/search') {
      const q = (url.searchParams.get('company') || '').toLowerCase();
      return json(db.invoices.filter(i => (i.company || '').toLowerCase().includes(q)));
    }

    if (p === '/api/invoices/upload' && method === 'POST') {
      const file = opts.body.get('file');
      let result = null, ms = null;

      if (INFERENCE_URL) {
        try {
          const t0 = performance.now();
          const fd = new FormData();
          fd.append('file', file);
          const res = await realFetch(INFERENCE_URL + '/extract', { method: 'POST', body: fd });
          if (!res.ok) throw new Error(await res.text());
          const d = await res.json();
          const f = d.extracted_fields || {};
          const s = d.confidence_scores || {};
          result = {
            company: f.company || '', date: f.date || '', total: f.total || '', address: f.address || '',
            scores: { company: s.company || 0, date: s.date || 0, total: s.total || 0, address: s.address || 0 }
          };
          ms = Math.round(d.processing_time_ms || (performance.now() - t0));
        } catch (e) {
          console.warn('Inference API unreachable, using canned values:', e);
        }
      }

      if (!result) {
        await new Promise(r => setTimeout(r, 1400));
        const keys = Object.keys(REAL_RESULTS);
        const key = REAL_RESULTS[stem(file.name)] ? stem(file.name) : keys[hash(file.name) % keys.length];
        result = REAL_RESULTS[key];
      }

      db.counter++;
      const inv = makeInvoice(db.counter, file.name, result);
      if (ms) inv.processing_time = ms;
      inv.pipeline_doc_id = inv.id;
      db.invoices.push(inv);
      save(db);
      return json(inv);
    }

    let m;
    if ((m = p.match(/^\/api\/invoices\/([^/]+)\/confirm$/)) && method === 'PUT') {
      const inv = db.invoices.find(i => i.id === m[1]);
      if (!inv) return json({ detail: 'not found' }, 404);
      const b = JSON.parse(opts.body || '{}');
      ['company', 'date', 'total', 'address'].forEach(k => { if (b[k] != null) inv[k] = b[k]; });
      inv.status = 'CONFIRMED';
      save(db);
      return json(inv);
    }

    if ((m = p.match(/^\/api\/invoices\/([^/]+)\/history$/))) {
      return json({ history: ['UPLOADED', 'OCR', 'LAYOUTLM', 'POST_PROCESS', 'SAVED']
        .map(stage => ({ stage, status: 'success' })) });
    }

    if ((m = p.match(/^\/api\/invoices\/([^/]+)$/)) && method === 'DELETE') {
      db.invoices = db.invoices.filter(i => i.id !== m[1]);
      save(db);
      return json({ deleted: true });
    }

    return json({ detail: 'not found' }, 404);
  }

  const realFetch = window.fetch.bind(window);

  window.fetch = async function (input, opts = {}) {
    const url = typeof input === 'string' ? input : input.url;
    if (!url.startsWith(AWS_URL)) return realFetch(input, opts);   // not ours
    const path = url.slice(AWS_URL.length) || '/';

    if (MODE === 'aws') return realFetch(input, opts);
    if (demoActive) return handle(path, opts);

    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 4000);
      const res = await realFetch(input, { ...opts, signal: ctrl.signal });
      clearTimeout(t);
      if (res.status >= 500) throw new Error('aws down');
      return res;
    } catch {
      demoActive = true;
      showBanner();
      return handle(path, opts);
    }
  };

  function showBanner() {
    if (document.getElementById('demo-banner') || !document.body) return;
    const b = document.createElement('div');
    b.id = 'demo-banner';
      'text-align:center;padding:7px;font:500 12px "JetBrains Mono",monospace;';
      '<a href="#" id="demo-reset" style="color:inherit;text-decoration:underline">reset demo data</a>';
    document.body.prepend(b);
    document.getElementById('demo-reset').onclick = (e) => {
      e.preventDefault(); localStorage.removeItem(DB_KEY); location.reload();
    };
    const txt = document.getElementById('statusText');
    const dot = document.getElementById('statusDot');
    const apply = () => {
      const t = txt.textContent;
      if (t === 'backend online') txt.textContent = INFERENCE_URL ? 'model online' : 'demo mode';
      else if (t === 'offline') txt.textContent = 'model offline';
    };
    apply();
    new MutationObserver(apply).observe(txt, { childList: true, characterData: true, subtree: true });
  }

  document.addEventListener('DOMContentLoaded', () => { if (demoActive) showBanner(); });
})();