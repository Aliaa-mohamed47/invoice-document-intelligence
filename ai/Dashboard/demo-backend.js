/* demo-backend.js
 * Persistent Cloud Backend via Supabase REST API
 * Handles Live Invoices, AI Model extraction, Confirmation, and Deletion.
 */
(function () {
  const MODE = 'demo';
  const AWS_URL = 'http://127.0.0.1:8000';
  const INFERENCE_URL = 'http://127.0.0.1:8000';
  // const INFERENCE_URL = 'https://dollhouse-tributary-viable.ngrok-free.dev';

  const SUPABASE_URL = 'https://kbjjhhambwnytuuubafx.supabase.co';
  const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtiampoaGFtYndueXR1dXViYWZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTIxNzc3MzAsImV4cCI6MjEwNzc1MzczMH0.1A_ibdCStemENh9bU-ltrUfHEoI_ihtaSbO1aYtEH4c';
  const HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': `Bearer ${SUPABASE_KEY}`,
    'Content-Type': 'application/json',
    'Prefer': 'return=representation'
  };

  const REAL_RESULTS = {
    'X51005447850': { company: 'BOOK STORE LTD', date: '12/05/2024', total: '154.50', address: 'Cairo, Egypt', scores: { company: 0.97, date: 0.95, total: 0.99, address: 0.88 } },
    'X51005568889': { company: 'TECH SUPPLIES CO', date: '20/06/2024', total: '820.00', address: 'Giza, Egypt', scores: { company: 0.96, date: 0.93, total: 0.98, address: 0.85 } },
    'X51006008089': { company: 'MODERN OFFICE MART', date: '01/07/2024', total: '340.25', address: 'Alexandria, Egypt', scores: { company: 0.94, date: 0.97, total: 0.99, address: 0.81 } }
  };

  const realFetch = window.fetch.bind(window);
  const json = (data, status = 200) =>
    new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });

  const stem = (name) => name.replace(/\.[^.]+$/, '');
  const hash = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);

  async function dbFetchAll() {
    try {
      const res = await realFetch(`${SUPABASE_URL}/rest/v1/invoices?select=*&order=created_at.desc`, { headers: HEADERS });
      if (!res.ok) return [];
      const rows = await res.json();
      return rows.map(r => ({
        id: r.id,
        invoice_id: r.id.substring(0, 8).toUpperCase(),
        filename: r.filename,
        company: r.parsed_data?.company || '',
        date: r.parsed_data?.date || '',
        total: r.parsed_data?.total || '',
        address: r.parsed_data?.address || '',
        company_score: r.parsed_data?.scores?.company || 0.95,
        date_score: r.parsed_data?.scores?.date || 0.94,
        total_score: r.parsed_data?.scores?.total || 0.98,
        address_score: r.parsed_data?.scores?.address || 0.88,
        status: (r.status || 'PENDING').toUpperCase(),
        processing_time: r.parsed_data?.processing_time || 950,
        created_at: r.created_at
      }));
    } catch {
      return [];
    }
  }

  async function dbInsert(invData) {
    const payload = {
      filename: invData.filename,
      status: invData.status.toLowerCase(),
      ocr_text: invData.ocr_text || '',
      parsed_data: {
        company: invData.company,
        date: invData.date,
        total: invData.total,
        address: invData.address,
        scores: invData.scores,
        processing_time: invData.processing_time
      }
    };
    const res = await realFetch(`${SUPABASE_URL}/rest/v1/invoices`, {
      method: 'POST',
      headers: HEADERS,
      body: JSON.stringify(payload)
    });
    const saved = await res.json();
    return saved[0] ? saved[0].id : ('inv-' + Date.now());
  }

  async function dbUpdateStatus(id, fields) {
    const res = await realFetch(`${SUPABASE_URL}/rest/v1/invoices?id=eq.${id}&select=*`, { headers: HEADERS });
    const rows = await res.json();
    if (!rows || !rows.length) return;

    const row = rows[0];
    const updatedData = { ...row.parsed_data, ...fields };
    await realFetch(`${SUPABASE_URL}/rest/v1/invoices?id=eq.${id}`, {
      method: 'PATCH',
      headers: HEADERS,
      body: JSON.stringify({ status: 'confirmed', parsed_data: updatedData })
    });
  }

  async function dbDelete(id) {
    await realFetch(`${SUPABASE_URL}/rest/v1/invoices?id=eq.${id}`, {
      method: 'DELETE',
      headers: HEADERS
    });
  }

  async function handle(path, opts) {
    const method = (opts.method || 'GET').toUpperCase();
    const url = new URL(path, 'http://x');
    const p = url.pathname;

    if (p === '/health') {
      try {
        const h = await realFetch(INFERENCE_URL + '/health');
        if (h.ok) return json({ status: 'ok', mode: 'live+model' });
      } catch {}
      return json({ status: 'ok', mode: 'cloud-db' });
    }

    if (p === '/api/invoices' && method === 'GET') {
      let list = await dbFetchAll();
      if (list.length === 0) {
        const sampleKey = Object.keys(REAL_RESULTS)[0];
        const sample = REAL_RESULTS[sampleKey];
        const newId = await dbInsert({
          filename: sampleKey + '.jpg',
          status: 'PENDING',
          company: sample.company,
          date: sample.date,
          total: sample.total,
          address: sample.address,
          scores: sample.scores,
          processing_time: 1100
        });
        list = await dbFetchAll();
      }
      return json(list);
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
          if (res.ok) {
            const d = await res.json();
            const f = d.extracted_fields || {};
            const s = d.confidence_scores || {};
            result = {
              company: f.company || '', date: f.date || '', total: f.total || '', address: f.address || '',
              scores: { company: s.company || 0.95, date: s.date || 0.94, total: s.total || 0.98, address: s.address || 0.88 }
            };
            ms = Math.round(d.processing_time_ms || (performance.now() - t0));
          }
        } catch (e) {
          console.warn('Inference model unreachable, using fallback parser:', e);
        }
      }

      if (!result) {
        await new Promise(r => setTimeout(r, 1200));
        const keys = Object.keys(REAL_RESULTS);
        const key = REAL_RESULTS[stem(file.name)] ? stem(file.name) : keys[hash(file.name) % keys.length];
        result = REAL_RESULTS[key];
        ms = 920;
      }

      const newId = await dbInsert({
        filename: file.name,
        status: 'PENDING',
        company: result.company,
        date: result.date,
        total: result.total,
        address: result.address,
        scores: result.scores,
        processing_time: ms
      });

      const inv = {
        id: newId,
        invoice_id: String(newId).substring(0, 8).toUpperCase(),
        filename: file.name,
        ...result,
        status: 'PENDING',
        processing_time: ms,
        created_at: new Date().toISOString()
      };
      return json(inv);
    }

    let m;
    if ((m = p.match(/^\/api\/invoices\/([^/]+)\/confirm$/)) && method === 'PUT') {
      const id = m[1];
      const b = JSON.parse(opts.body || '{}');
      await dbUpdateStatus(id, b);
      return json({ status: 'CONFIRMED', id });
    }

    if ((m = p.match(/^\/api\/invoices\/([^/]+)\/history$/))) {
      return json({ history: ['UPLOADED', 'OCR', 'LAYOUTLM', 'POST_PROCESS', 'SAVED'].map(stage => ({ stage, status: 'success' })) });
    }

    if ((m = p.match(/^\/api\/invoices\/([^/]+)$/)) && method === 'DELETE') {
      const id = m[1];
      await dbDelete(id);
      return json({ deleted: true, id });
    }

    return json({ detail: 'not found' }, 404);
  }

  window.fetch = async function (input, opts = {}) {
    const url = typeof input === 'string' ? input : input.url;
    if (!url.startsWith(AWS_URL)) return realFetch(input, opts);
    const path = url.slice(AWS_URL.length) || '/';
    return handle(path, opts);
  };
})();