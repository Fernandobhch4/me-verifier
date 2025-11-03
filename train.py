import numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump
X = np.load("reports/embeddings.npy")
y = np.loadtxt("reports/labels.csv", delimiter=",", skiprows=1, usecols=[0])
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(Xtr)
Xtr, Xval = scaler.transform(Xtr), scaler.transform(Xval)
clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
pred = clf.predict(Xval)
proba = clf.predict_proba(Xval)[:,1]
metrics = {"acc": float(accuracy_score(yval,pred)), "auc": float(roc_auc_score(yval,proba))}
dump(clf, "models/model.joblib"); dump(scaler, "models/scaler.joblib")
with open("reports/metrics.json","w") as f: json.dump(metrics,f,indent=2)
print(metrics)
