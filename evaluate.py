import numpy as np, json
from sklearn.metrics import precision_recall_curve, roc_curve
from joblib import load
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.load("reports/embeddings.npy")
y = np.loadtxt("reports/labels.csv", delimiter=",", skiprows=1, usecols=[0])

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = load("models/scaler.joblib")
clf    = load("models/model.joblib")
probs  = clf.predict_proba(scaler.transform(Xva))[:,1]

prec, rec, thr = precision_recall_curve(yva, probs)
f1 = (2*prec*rec)/(prec+rec+1e-9)
best = int(f1.argmax())
tau  = float(thr[max(best-1,0)])
json.dump({"threshold": tau}, open("reports/threshold.json","w"), indent=2)
print("Umbral óptimo τ* =", tau)

fpr, tpr, _ = roc_curve(yva, probs)
plt.plot(fpr, tpr)
plt.title("Curva ROC")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.savefig("reports/roc.png", dpi=140)
