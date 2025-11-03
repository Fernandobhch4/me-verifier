from facenet_pytorch import InceptionResnetV1
from PIL import Image
from pathlib import Path
import torch, numpy as np, pandas as pd

def embed_dir(model, folder, label, device):
    xs, ys = [], []
    for p in Path(folder).glob("*"):
        if p.suffix.lower() not in [".jpg",".jpeg",".png"]: continue
        img = Image.open(p).convert("RGB")
        t = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
        t = (t - 0.5) / 0.5
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(t).squeeze(0).cpu().numpy()
        xs.append(emb); ys.append(label)
    return np.array(xs), np.array(ys)

if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    x_me, y_me = embed_dir(model, "data/cropped/me", 1, device)
    x_not, y_not = embed_dir(model, "data/cropped/not_me", 0, device)
    X = np.vstack([x_me, x_not]); y = np.concatenate([y_me, y_not])
    np.save("reports/embeddings.npy", X)
    pd.DataFrame({"label": y}).to_csv("reports/labels.csv", index=False)
