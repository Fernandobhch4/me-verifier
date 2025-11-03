from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
import torch

SRC_DIRS = [("data/me", "data/cropped/me"), ("data/not_me", "data/cropped/not_me")]

def crop_folder(src, dst, image_size=160):
    mtcnn = MTCNN(image_size=image_size, margin=20, post_process=True, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    Path(dst).mkdir(parents=True, exist_ok=True)
    for p in Path(src).glob("*"):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            img = Image.open(p).convert("RGB")
            face = mtcnn(img)
            if face is None:
                continue
            out = Image.fromarray((face.permute(1,2,0).clamp(0,1).mul(255).byte().cpu().numpy()))
            out.save(Path(dst)/p.name)
        except Exception as e:
            print(f"Error en {p.name}: {e}")

if __name__ == "__main__":
    for src, dst in SRC_DIRS:
        crop_folder(src, dst)
