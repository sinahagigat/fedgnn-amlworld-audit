"""non-graph reference models on the AMLworld split, mirroring the Elliptic references of paper 1: random forest (200 trees, twenty seeds), histogram-based gradient-boosted trees (300 iterations, learning rate 0.05, no early stopping) and logistic regression (L2, C=1, lbfgs, max_iter=5000), on standardized node features, fixed hyperparameters, no class weighting, scikit-learn defaults otherwise

    python analysis/aml_references.py [--data-dir <local-path>] [--out analysis/output]
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEEDS = list(range(42, 62))
FIELDS = ["model", "seed", "test_ap", "test_auc", "val_ap", "val_auc", "fit_sec", "n_train", "n_features", "sklearn"]


def load(data_dir):
    marker = Path(data_dir) / "AMLWORLD"
    graph = Path(data_dir) / marker.read_text(encoding="utf-8").strip()
    d = torch.load(graph, weights_only=False)
    X = d.x.cpu().numpy().astype(np.float32)
    y = d.y.cpu().numpy().astype(np.int64)
    m = {k: getattr(d, f"{k}_mask").cpu().numpy().astype(bool) for k in ("train", "val", "test")}
    for k, v in m.items():
        assert set(np.unique(y[v])) <= {0, 1}, f"unexpected labels in {k} mask"
    print(f"graph {graph.name}: {X.shape[0]:,} nodes, {X.shape[1]} features | "
          + " | ".join(f"{k} {v.sum():,} (pos {y[v].mean():.4%})" for k, v in m.items()), flush=True)
    sc = StandardScaler().fit(X[m["train"]])
    return sc.transform(X).astype(np.float32), y, m


def score(model, X, y, mask):
    p = model.predict_proba(X[mask])[:, 1]
    return average_precision_score(y[mask], p), roc_auc_score(y[mask], p)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"<local-path>")
    ap.add_argument("--out", default=str(here / "output"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "references.csv"
    done = set()
    if path.is_file():
        with open(path, newline="") as f:
            done = {(r["model"], int(r["seed"])) for r in csv.DictReader(f)}

    X, y, m = load(a.data_dir)
    d = X.shape[1]
    plan = [("Logistic regression", 42, lambda s: LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)),
            ("Gradient-boosted trees", 42, lambda s: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, early_stopping=False, random_state=s))]
    plan += [("Random forest", s, lambda s: RandomForestClassifier(
        n_estimators=200, max_features=min(50, d), n_jobs=a.jobs, random_state=s)) for s in SEEDS]

    new = not path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, (name, seed, make) in enumerate(plan, 1):
            if (name, seed) in done:
                continue
            t0 = time.time()
            model = make(seed).fit(X[m["train"]], y[m["train"]])
            sec = time.time() - t0
            tap, tauc = score(model, X, y, m["test"]); vap, vauc = score(model, X, y, m["val"])
            w.writerow({"model": name, "seed": seed, "test_ap": f"{tap:.6f}", "test_auc": f"{tauc:.6f}",
                        "val_ap": f"{vap:.6f}", "val_auc": f"{vauc:.6f}", "fit_sec": f"{sec:.1f}",
                        "n_train": int(m["train"].sum()), "n_features": d, "sklearn": sklearn.__version__})
            f.flush()
            print(f"[{i}/{len(plan)}] {name} seed {seed}: test AP {tap:.4f} | val AP {vap:.4f} | {sec / 60:.1f} min", flush=True)
            del model
    print(f"done -> {path}")


if __name__ == "__main__":
    main()
