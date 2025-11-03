# scripts/crop_faces.py
import os
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# Entradas y salida
INPUTS = [("data/me", "me"), ("data/not_me", "not_me")]
OUT_ROOT = Path("data/cropped")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)

def main():
    print(f"[INFO] Device: {device}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0

    for in_dir, label in INPUTS:
        src = Path(in_dir)
        dst = OUT_ROOT / label
        dst.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[WARN] No existe {src}, se omite.")
            continue

        saved = 0
        imgs = list(src.rglob("*.*"))  # soporta subcarpetas
        print(f"[INFO] {label}: {len(imgs)} imágenes encontradas en {src}")

        for i, p in enumerate(imgs, 1):
            try:
                img = Image.open(p).convert("RGB")
                face = mtcnn(img)
                if face is None:
                    # no se detectó rostro
                    continue
                face_img = face.permute(1, 2, 0).byte().numpy()
                # evitar nombres duplicados
                out_name = (dst / p.stem).with_suffix(".png")
                out_name = out_name if not out_name.exists() else (dst / f"{p.stem}_{i}.png")
                Image.fromarray(face_img).save(out_name)
                saved += 1
                if saved % 50 == 0:
                    print(f"[{label}] guardadas: {saved}")
            except Exception as e:
                print(f"[WARN] {p.name}: {e}")

        total += saved
        print(f"[OK] {label}: {saved} rostros recortados → {dst}")

    print(f"[DONE] Total recortes: {total}")

if __name__ == "__main__":
    main()
