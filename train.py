import numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump
from pathlib import Path

X = np.load("reports/embeddings.npy")
y = np.loadtxt("reports/labels.csv", delimiter=",", skiprows=1, usecols=[0])

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

clf = LogisticRegression(max_iter=200)
clf.fit(Xtr_s, ytr)
pred = clf.predict_proba(Xva_s)[:,1]

metrics = {
    "accuracy": float(accuracy_score(yva, pred>=0.5)),
    "roc_auc": float(roc_auc_score(yva, pred))
}
Path("models").mkdir(exist_ok=True)
dump(clf, "models/model.joblib"); dump(scaler, "models/scaler.joblib")
Path("reports").mkdir(exist_ok=True)
json.dump(metrics, open("reports/metrics.json","w"), indent=2)
print(metrics)
