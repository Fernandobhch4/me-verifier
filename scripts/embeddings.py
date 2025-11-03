# scripts/embeddings.py
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

CROPPED = Path("data/cropped")
REPORTS = Path("reports")
BATCH = 32

def load_images_with_labels():
    paths, labels = [], []
    for label_name, label_val in [("me", 1), ("not_me", 0)]:
        cls_dir = CROPPED / label_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.png"):
            paths.append(p)
            labels.append(label_val)
    return paths, np.array(labels, dtype=np.int64)

def to_tensor(img: Image.Image):
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    return fixed_image_standardization(t)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    paths, y = load_images_with_labels()
    n = len(paths)
    if n == 0:
        print("[ERROR] No hay imágenes en data/cropped. Corre primero crop_faces.py")
        return

    print(f"[INFO] Imágenes totales: {n} (me={np.sum(y==1)}, not_me={np.sum(y==0)})")

    REPORTS.mkdir(parents=True, exist_ok=True)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    X = []
    for i in range(0, n, BATCH):
        batch_paths = paths[i:i+BATCH]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(to_tensor(img))
            except Exception as e:
                print(f"[WARN] {p.name}: {e}")

        if not batch_imgs:
            continue

        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = embedder(batch).cpu().numpy()  # (B, 512)
        X.append(emb)

        if (i // BATCH) % 10 == 0:
            print(f"[PROG] {min(i+BATCH, n)}/{n}")

    X = np.vstack(X)
    np.save(REPORTS / "embeddings.npy", X)
    pd.DataFrame({"label": y}).to_csv(REPORTS / "labels.csv", index=False)

    print(f"[DONE] Guardado: reports/embeddings.npy shape={X.shape}")
    print(f"[DONE] Guardado: reports/labels.csv N={len(y)}")

if __name__ == "__main__":
    main()
