# api/app.py
import io, time, os
from PIL import Image
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from joblib import load
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch, numpy as np

load_dotenv()

THRESHOLD  = float(os.getenv("THRESHOLD", "0.75"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH= os.getenv("SCALER_PATH", "models/scaler.joblib")
MAX_MB     = int(os.getenv("MAX_MB", "5"))

app = Flask(__name__, static_folder="static", template_folder="templates")

device   = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn    = MTCNN(image_size=160, margin=20, post_process=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
clf      = load(MODEL_PATH)
scaler   = load(SCALER_PATH)

@app.get("/")
def home():
    # Página simple para subir imagen y ver el resultado
    return render_template("index.html", threshold=THRESHOLD)

@app.get("/healthz")
def healthz():
    return {"status":"ok","model":"me-verifier-v1","threshold":THRESHOLD}

@app.post("/verify")
def verify():
    t0 = time.time()
    f = request.files.get("image")
    if not f:
        return jsonify(error="falta 'image'"), 400
    if f.mimetype not in ("image/jpeg","image/png"):
        return jsonify(error="solo image/jpeg o image/png"), 415

    data = f.read()
    if len(data) > MAX_MB*1024*1024:
        return jsonify(error=f"archivo > {MAX_MB}MB"), 413

    img = Image.open(io.BytesIO(data)).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return jsonify(error="no se detectó rostro"), 422

    with torch.no_grad():
        emb = embedder(face.unsqueeze(0).to(device)).cpu().numpy()

    x     = scaler.transform(emb)
    score = float(clf.predict_proba(x)[0,1]) if hasattr(clf,"predict_proba") else float(clf.decision_function(x)[0])
    is_me = (score >= THRESHOLD)
    return jsonify(
        is_me=bool(is_me),
        score=round(score,4),
        threshold=THRESHOLD,
        timing_ms=round((time.time()-t0)*1000,1)
    )
