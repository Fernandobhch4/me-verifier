import io, os, time
from PIL import Image
from flask import Flask, request, jsonify, Response
from joblib import load
from dotenv import load_dotenv

from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np

# === Config ===
load_dotenv()
MODEL_PATH  = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
THRESHOLD   = float(os.getenv("THRESHOLD", "0.75"))
MAX_MB      = int(os.getenv("MAX_MB", "5"))

# === App ===
app = Flask(__name__)

# === Modelos / Pipelines ===
device   = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn    = MTCNN(image_size=160, margin=20, post_process=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
clf      = load(MODEL_PATH)
scaler   = load(SCALER_PATH)

# === Helpers ===
def face_embedding(img_pil):
    face = mtcnn(img_pil)
    if face is None:
        return None
    t = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedder(t).squeeze(0).cpu().numpy()
    return emb

# === Rutas API ===
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/verify")
def verify():
    t0 = time.time()

    if "image" not in request.files:
        return jsonify(error="imagen requerida (campo 'image')"), 400

    f = request.files["image"]
    if f.mimetype not in ["image/jpeg", "image/png"]:
        return jsonify(error="solo image/jpeg o image/png"), 415

    # tamaño
    f.seek(0, io.SEEK_END)
    size_mb = f.tell() / (1024 * 1024)
    if size_mb > MAX_MB:
        return jsonify(error=f"archivo > {MAX_MB}MB"), 413
    f.seek(0)

    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify(error="archivo inválido"), 400

    emb = face_embedding(img)
    if emb is None:
        return jsonify(error="no se detectó rostro"), 422

    prob = float(clf.predict_proba(scaler.transform([emb]))[0, 1])
    is_me = prob >= THRESHOLD

    resp = {
        "model_version": "me-verifier-v1",
        "is_me": bool(is_me),
        "score": round(prob, 4),
        "threshold": THRESHOLD,
        "timing_ms": round((time.time() - t0) * 1000, 2)
    }
    return jsonify(resp), 200

# === Vista única (UI) ===
@app.get("/")
def single_page():
    html = f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Me Verifier</title>
<style>
  :root {{
    --bg: #0b0c10; --panel:#11131a; --panel2:#0f1117; --stroke:#1f2330;
    --text:#e6eef8; --muted:#97a4b8; --primary:#5b6bff; --primary-press:#4b59e6;
    --good:#22c55e; --bad:#ef4444; --warn:#f59e0b;
    --radius:16px;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg:#f7f9fc; --panel:#ffffff; --panel2:#ffffff; --stroke:#e7ecf4;
      --text:#0f172a; --muted:#536076; --primary:#3b5bff;
    }}
  }}
  * {{ box-sizing:border-box }}
  body {{
    margin:0; background:var(--bg); color:var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
  }}
  .wrap {{ min-height:100dvh; display:grid; place-items:center; padding:40px 16px; }}
  .card {{
    width: 100%; max-width: 920px; background:var(--panel); border:1px solid var(--stroke);
    border-radius: var(--radius); box-shadow: 0 20px 40px rgba(0,0,0,.25);
    padding: 24px 24px 20px;
  }}
  header {{
    display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:12px;
  }}
  h1 {{ font-size:22px; margin:0 }}
  .status {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-weight:600; }}
  .dot {{ width:10px; height:10px; border-radius:50%; background:var(--warn); box-shadow:0 0 0 3px rgba(0,0,0,.08) inset; }}
  .dot.ok {{ background:var(--good) }}
  .dot.err{{ background:var(--bad) }}

  .desc {{ margin: 4px 0 20px; color:var(--muted) }}

  .grid {{ display:grid; grid-template-columns: 1.1fr 1fr; gap: 18px; }}
  @media (max-width: 880px) {{ .grid {{ grid-template-columns: 1fr; }} }}

  .drop {{
    border: 2px dashed var(--stroke); background: var(--panel2);
    border-radius: var(--radius); padding: 18px; text-align:center; cursor:pointer;
    min-height: 210px; display:flex; flex-direction:column; justify-content:center; gap:10px;
  }}
  .drop:hover {{ border-color: var(--primary); }}
  .drop strong {{ font-size:15px }}
  .hint {{ color:var(--muted); font-size:12px }}
  input[type=file] {{ display:none; }}
  img#preview {{ max-width: 260px; max-height: 200px; border-radius: 12px; margin: 4px auto 0; display:none; }}

  .controls {{ display:flex; gap:10px; margin: 2px 0 12px; flex-wrap:wrap; }}
  .btn {{
    border:1px solid transparent; border-radius:12px; padding:10px 16px; font-weight:700; cursor:pointer;
    background:var(--primary); color:#fff;
  }}
  .btn:disabled {{ opacity:.65; cursor:not-allowed; }}
  .btn:hover {{ background: var(--primary-press); }}
  .btn.outline {{
    background:transparent; color:var(--text); border-color:var(--stroke);
  }}
  .btn.outline:hover {{ border-color:var(--primary); color:var(--primary); }}

  .panel {{
    background:var(--panel2); border:1px solid var(--stroke); border-radius: var(--radius); padding: 14px;
  }}

  .result {{
    display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px;
  }}
  .badge {{
    display:inline-flex; align-items:center; gap:8px;
    border-radius:999px; padding:8px 12px; font-weight:800; letter-spacing:.2px;
  }}
  .badge.good {{ background: rgba(34,197,94,.12); color: var(--good); border:1px solid rgba(34,197,94,.35); }}
  .badge.bad  {{ background: rgba(239,68,68,.12); color: var(--bad);  border:1px solid rgba(239,68,68,.35); }}

  .kv {{ display:grid; grid-template-columns: auto 1fr; gap:6px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .kv div:nth-child(odd) {{ color:var(--muted) }}

  .footer-note {{ color:var(--muted); font-size:12px; margin-top:10px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <header>
      <h1>Me Verifier</h1>
      <div class="status"><span id="dot" class="dot"></span><span id="svc">Comprobando servicio…</span></div>
    </header>
    <p class="desc">Sube una imagen (JPEG/PNG ≤ {MAX_MB} MB) y verifica si eres tú. Todo se procesa localmente en este servidor.</p>

    <div class="grid">
      <!-- IZQUIERDA: Dropzone -->
      <div>
        <label for="file" class="drop" id="drop">
          <strong>Haz click o suelta aquí tu imagen</strong>
          <div class="hint">Formatos permitidos: JPEG/PNG</div>
          <input id="file" name="image" type="file" accept="image/jpeg,image/png" required>
          <img id="preview" alt="preview" />
        </label>
        <div class="controls">
          <button id="btn" class="btn" type="button">Verificar</button>
          <button id="clear" class="btn outline" type="button">Limpiar</button>
        </div>
      </div>

      <!-- DERECHA: Resultado -->
      <div class="panel">
        <div class="result">
          <div id="badge" class="badge" style="display:none">Resultado</div>
          <div class="hint" id="timing">Latencia: — ms</div>
        </div>
        <div class="kv">
          <div>Score:</div>     <div id="score">—</div>
          <div>Threshold:</div> <div id="th">—</div>
          <div>Modelo:</div>    <div id="ver">—</div>
        </div>
        <div class="footer-note">Endpoints: <code>GET /healthz</code> · <code>POST /verify</code></div>
      </div>
    </div>
  </div>
</div>

<script>
const file   = document.getElementById('file');
const drop   = document.getElementById('drop');
const btn    = document.getElementById('btn');
const clearB = document.getElementById('clear');
const prev   = document.getElementById('preview');

const badge  = document.getElementById('badge');
const timing = document.getElementById('timing');
const score  = document.getElementById('score');
const th     = document.getElementById('th');
const ver    = document.getElementById('ver');

const dot    = document.getElementById('dot');
const svc    = document.getElementById('svc');

function setService(ok) {{
  dot.classList.remove('ok','err');
  if (ok) {{ dot.classList.add('ok'); svc.textContent = 'Servicio operativo'; }}
  else     {{ dot.classList.add('err'); svc.textContent = 'Servicio caído'; }}
}}

async function ping() {{
  try {{
    const r = await fetch('/healthz');
    setService(r.ok);
  }} catch {{ setService(false); }}
}}
ping();

function previewImage(f) {{
  if (!f) {{ prev.style.display='none'; prev.src=''; return; }}
  const url = URL.createObjectURL(f);
  prev.src = url; prev.style.display='block';
}}

drop.addEventListener('click', () => file.click());
drop.addEventListener('dragover', e => {{ e.preventDefault(); drop.style.borderColor='#5b6bff'; }});
drop.addEventListener('dragleave', () => drop.style.borderColor='var(--stroke)');
drop.addEventListener('drop', e => {{
  e.preventDefault();
  drop.style.borderColor='var(--stroke)';
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {{
    file.files = e.dataTransfer.files;
    previewImage(file.files[0]);
  }}
}});
file.addEventListener('change', () => previewImage(file.files[0]));

function resetUI() {{
  badge.style.display='none';
  badge.className='badge';
  badge.textContent='Resultado';
  timing.textContent='Latencia: — ms';
  score.textContent='—';
  th.textContent='—';
  ver.textContent='—';
}}

clearB.addEventListener('click', () => {{
  file.value = '';
  previewImage(null);
  resetUI();
}});

async function doVerify() {{
  if (!file.files[0]) {{
    badge.style.display='inline-flex';
    badge.className='badge bad';
    badge.textContent='Falta imagen';
    return;
  }}
  btn.disabled = true; badge.style.display='inline-flex'; badge.className='badge'; badge.textContent='Verificando…';
  try {{
    const fd = new FormData();
    fd.append('image', file.files[0]);
    const r = await fetch('/verify', {{ method:'POST', body:fd }});
    const data = await r.json();
    if (!r.ok) {{
      badge.className='badge bad';
      badge.textContent = data.error || 'Error';
      timing.textContent = 'Latencia: — ms';
      score.textContent='—'; th.textContent='—'; ver.textContent='—';
      return;
    }}
    // OK
    badge.className = 'badge ' + (data.is_me ? 'good' : 'bad');
    badge.textContent = data.is_me ? 'Soy yo' : 'No soy yo';
    timing.textContent = 'Latencia: ' + (data.timing_ms ?? '—') + ' ms';
    score.textContent  = (data.score ?? '—');
    th.textContent     = (data.threshold ?? '—');
    ver.textContent    = (data.model_version ?? '—');
  }} catch (err) {{
    badge.className='badge bad';
    badge.textContent='Error de red';
  }} finally {{
    btn.disabled = false;
  }}
}}

btn.addEventListener('click', doVerify);
</script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")

# === Main ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
