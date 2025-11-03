import io, os, time
from PIL import Image
from flask import Flask, request, jsonify, Response
from joblib import load
from dotenv import load_dotenv
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np

load_dotenv()
MODEL_PATH  = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
THRESHOLD   = float(os.getenv("THRESHOLD", "0.75"))
MAX_MB      = int(os.getenv("MAX_MB", "5"))

app = Flask(__name__)

device   = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn    = MTCNN(image_size=160, margin=20, post_process=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
clf      = load(MODEL_PATH)
scaler   = load(SCALER_PATH)

def face_embedding(img_pil):
    face = mtcnn(img_pil)
    if face is None:
        return None
    t = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedder(t).squeeze(0).cpu().numpy()
    return emb

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

@app.get("/")
def ui():
    return Response(
        "<h2>Me Verifier</h2><p>Interfaz disponible en versión unificada (HTML + JS).</p>",
        mimetype="text/html"
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
