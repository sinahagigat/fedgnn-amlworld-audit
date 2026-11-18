"""E1 (FED-PUB on Elliptic) analysis: every FED-PUB number of the manuscript, reproducible

    python analysis/e1_analysis.py --e1-results results
        --paper-results <fedgnn-elliptic-audit results>
        --frozen-repo <fedgnn-elliptic-audit>
        [--out analysis/output]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEDPUB, FEDPUB_W = "FedPub_GCN", "FedPub_GCN_wloss"
FEDPUB_L, FEDPUB_O = "FedPub_GCN_lr1e3", "FedPub_GCN_local"
FEDPUB_ALL = (FEDPUB, FEDPUB_W, FEDPUB_L, FEDPUB_O)
SHORT = {"FedAvg_GCN": "FedAvg", "FedProx_GCN": "FedProx", "FedProto_GCN": "FedProto",
         "FedGNN_Hard_GCN": "Hard", "FedGNN_Soft_GCN": "Soft", "FedGNN_Soft_Virtual_GCN": "Soft-VN",
         "FedGNN_Soft_Virtual_GAT": "Soft-VN-GAT", "FedAvg_MLP": "FedAvg-MLP",
         FEDPUB: "FED-PUB", FEDPUB_W: "FED-PUB (weighted loss)",
         FEDPUB_L: "FED-PUB (lr 1e-3)", FEDPUB_O: "Local (FED-PUB model, no communication)"}
N = {}


# loading
def protocol_of(job):
    if job[:2] in ("PA",) or (job[0] == "A" and job[1:].isdigit()):
        return "A"
    if job[:2] == "PN" or (job[0] in "NM" and job[1:].isdigit()):
        return "N"
    if job[:2] in ("PW", "PL", "PO"):
        return "N"  # same protocol, different variant name
    if job[:2] == "PQ" and len(job) == 4:
        return "Q" + job[2]
    if job[0] == "Q" and "_" in job:
        return "Q" + job[1:job.index("_")]
    return None


def fingerprint(r):
    pc = (r.get("partition_diagnostics") or {}).get("per_client") or {}
    return tuple((int(v["num_nodes"]), int(v["num_train"]), int(v["num_intra_edges"]))
                 for _, v in sorted(pc.items(), key=lambda kv: int(kv[0]))) or None


def read_dir(d, source):
    f = d / "artifacts.json"
    if not f.is_file():
        f = d / "artifacts_slim.json"
    if not f.is_file():
        return [], None
    proto = protocol_of(d.name)
    if proto is None:
        return [], None
    J = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for r in J["runs"].values():
        if r.get("status") != "completed" or r.get("algorithm") == "centralized":
            continue
        loc = r.get("test_metrics_local") or {}
        les = r.get("test_metrics_local_es") or {}
        rows.append(dict(protocol=proto, job=d.name, source=source, variant=r["variant_name"],
                         seed=int(r["seed"]), ap_local=loc.get("ap", np.nan),
                         ap_local_es=les.get("ap", np.nan), auc_local=loc.get("auc", np.nan),
                         best_val_ap=r.get("best_val_ap", np.nan), best_round=r.get("best_round_idx", np.nan),
                         es_stop=r.get("es_stop_round", np.nan), loss=r.get("fedpub_loss", ""),
                         code_hash=r.get("code_hash", J.get("code_hash")),
                         config_hash=r.get("config_hash", J.get("config_hash")),
                         fp=fingerprint(r)))
    return rows, (d.name, source, J.get("code_hash"), J.get("config_hash"), len(rows))


def load(e1, paper, frozen):
    rows, prov = [], []
    dirs = [(d, "e1") for d in Path(e1).iterdir() if d.is_dir() and d.name[:2] in ("PA", "PN", "PW", "PL", "PO", "PQ")]
    dirs += [(d, "paper") for d in Path(paper).iterdir() if d.is_dir() and d.name in ("A1", "A2", "A3", "A4")]
    for d in Path(frozen).glob("*/results/v7/*"):
        if d.is_dir() and protocol_of(d.name) is not None and d.name[0] in "NMQ":
            dirs.append((d, "paper"))
    for d, src in sorted(dirs, key=lambda t: t[0].name):
        r, p = read_dir(d, src)
        rows += r
        if p:
            prov.append(p)
    df = pd.DataFrame(rows)
    dup = df.duplicated(["protocol", "variant", "seed"])
    assert not dup.any(), f"duplicate runs: {df[dup][['protocol', 'variant', 'seed']].values.tolist()}"
    return df, pd.DataFrame(prov, columns=["job", "source", "code_hash", "config_hash", "runs"])


# statistics (as e2_analysis.py)
def wil(x, y):
    d = np.asarray(x) - np.asarray(y)
    if len(d) < 2 or np.all(d == 0):
        return 1.0
    method = "exact" if len(d) <= 25 and np.all(d != 0) else "auto"
    return float(stats.wilcoxon(x, y, alternative="two-sided", method=method).pvalue)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); out = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); out[k] = min(run, 1.0)
    return out


def ci95(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean() - h, d.mean() + h


def cmp(a, b):
    s = a.dropna().index.intersection(b.dropna().index)
    x, y = a.loc[s].values, b.loc[s].values; d = x - y; lo, hi = ci95(d)
    return dict(n=len(d), diff=d.mean() if len(d) else np.nan, lo=lo, hi=hi,
                p=wil(x, y) if len(d) else np.nan, wins=int((d > 0).sum()))


def W(df, proto, col="ap_local"):
    return df[df.protocol == proto].pivot(index="seed", columns="variant", values=col)


def _md(t):
    try:
        return t.to_markdown(index=False)
    except ImportError:
        return "```\n" + t.to_string(index=False) + "\n```"


def f4(x): return "" if pd.isna(x) else f"{x:.4f}"
def s4(x): return "" if pd.isna(x) else f"{x:+.4f}"
def pf(p): return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")
def ci(c): return f"{s4(c['diff'])} [{s4(c['lo'])}, {s4(c['hi'])}]"


# checks
def partition_check(df):
    out = []
    for proto in sorted(df.protocol.unique()):
        sub = df[df.protocol == proto]
        ref = sub[sub.variant == "FedAvg_GCN"].set_index("seed").fp
        for v in FEDPUB_ALL:
            mine = sub[sub.variant == v].set_index("seed").fp
            s = mine.index.intersection(ref.index)
            ok = [mine[k] == ref[k] for k in s if mine[k] is not None and ref[k] is not None]
            if len(s):
                out.append(dict(protocol=proto, variant=SHORT[v], seeds=len(s), checked=len(ok),
                                identical=int(sum(ok))))
    t = pd.DataFrame(out)
    N["partition_check_all_identical"] = bool(len(t) and (t.checked == t.identical).all())
    return t


# tables
def table_protocols(df):
    """A and N: FED-PUB against FedAvg and Soft; Holm over the two protocols per comparison"""
    rows = []
    for proto, label in (("A", "A: redrawn Louvain, budget 1, early stopping"),
                         ("N", "N: redrawn Louvain, full budget, no early stopping")):
        w = W(df, proto)
        if FEDPUB not in w or "FedAvg_GCN" not in w:
            continue
        w = w[[FEDPUB, "FedAvg_GCN", "FedGNN_Soft_GCN"]].dropna()  # common seeds only
        a, s = cmp(w[FEDPUB], w["FedAvg_GCN"]), cmp(w[FEDPUB], w["FedGNN_Soft_GCN"])
        rows.append(dict(protocol=label, n=a["n"], FedAvg=w["FedAvg_GCN"].mean(), Soft=w["FedGNN_Soft_GCN"].mean(),
                         FEDPUB=w[FEDPUB].mean(), vs_FedAvg=a, vs_Soft=s))
    for key in ("vs_FedAvg", "vs_Soft"):
        hp = holm([r[key]["p"] for r in rows])
        for r, h in zip(rows, hp):
            r[key]["holm"] = h
    t = pd.DataFrame([dict(protocol=r["protocol"], n=r["n"], FedAvg=f4(r["FedAvg"]), Soft=f4(r["Soft"]),
                           **{"FED-PUB": f4(r["FEDPUB"]),
                              "FED-PUB minus FedAvg [95% CI]": ci(r["vs_FedAvg"]),
                              "FED-PUB wins": r["vs_FedAvg"]["wins"], "Holm p (FedAvg)": pf(r["vs_FedAvg"]["holm"]),
                              "FED-PUB minus Soft [95% CI]": ci(r["vs_Soft"]),
                              "Holm p (Soft)": pf(r["vs_Soft"]["holm"])}) for r in rows])
    for r in rows:
        k = r["protocol"][0]
        N[f"{k}_fedavg"], N[f"{k}_soft"], N[f"{k}_fedpub"] = r["FedAvg"], r["Soft"], r["FEDPUB"]
        N[f"{k}_fedpub_minus_fedavg"], N[f"{k}_fedpub_wins"] = r["vs_FedAvg"]["diff"], r["vs_FedAvg"]["wins"]
        N[f"{k}_fedpub_minus_fedavg_holm"] = r["vs_FedAvg"]["holm"]
        N[f"{k}_fedpub_minus_soft"], N[f"{k}_fedpub_minus_soft_holm"] = r["vs_Soft"]["diff"], r["vs_Soft"]["holm"]
    return t


def table_n_all(df):
    """protocol N: every federated design against FedAvg under per-client inference (Holm over designs)"""
    w = W(df, "N")
    order = [v for v in ("FedProx_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN",
                         "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT", "FedAvg_MLP", FEDPUB, FEDPUB_W)
             if v in w]
    cs = [cmp(w[v], w["FedAvg_GCN"]) for v in order]
    hp = holm([c["p"] for c in cs])
    rows = [{"design": "FedAvg", "n": int(w["FedAvg_GCN"].notna().sum()), "AP": f4(w["FedAvg_GCN"].mean()),
             "SD": f4(w["FedAvg_GCN"].std()), "minus FedAvg [95% CI]": "", "wins": "", "Holm p": ""}]
    for v, c, h in zip(order, cs, hp):
        rows.append({"design": SHORT[v], "n": c["n"], "AP": f4(w[v].mean()), "SD": f4(w[v].std()),
                     "minus FedAvg [95% CI]": ci(c), "wins": c["wins"], "Holm p": pf(h)})
    return pd.DataFrame(rows)


def table_fixed(df):
    """Q0-Q9: FedAvg, Soft and FED-PUB per fixed partition; Holm over the ten partitions"""
    rows = []
    for k in range(10):
        w = W(df, f"Q{k}")
        if FEDPUB not in w or "FedAvg_GCN" not in w:
            continue
        w = w[[FEDPUB, "FedAvg_GCN", "FedGNN_Soft_GCN"]].dropna()  # common seeds only
        a, s, sf = cmp(w[FEDPUB], w["FedAvg_GCN"]), cmp(w[FEDPUB], w["FedGNN_Soft_GCN"]), \
            cmp(w["FedGNN_Soft_GCN"], w["FedAvg_GCN"])
        rows.append(dict(k=k, n=a["n"], fa=w["FedAvg_GCN"].mean(), so=w["FedGNN_Soft_GCN"].mean(),
                         fp=w[FEDPUB].mean(), a=a, s=s, sf=sf))
    for key in ("a", "s", "sf"):
        for r, h in zip(rows, holm([r[key]["p"] for r in rows])):
            r[key]["holm"] = h
    t = pd.DataFrame([{"partition": f"Q{r['k']}", "n": r["n"], "FedAvg": f4(r["fa"]), "Soft": f4(r["so"]),
                       "FED-PUB": f4(r["fp"]), "FED-PUB minus FedAvg [95% CI]": ci(r["a"]),
                       "FED-PUB wins": r["a"]["wins"], "Holm p": pf(r["a"]["holm"]),
                       "FED-PUB minus Soft": s4(r["s"]["diff"]), "Holm p (Soft)": pf(r["s"]["holm"]),
                       "Soft minus FedAvg": s4(r["sf"]["diff"]), "Holm p (Soft-FedAvg)": pf(r["sf"]["holm"])}
                      for r in rows])
    if rows:
        fp = np.array([r["fp"] for r in rows]); fa = np.array([r["fa"] for r in rows])
        so = np.array([r["so"] for r in rows]); dd = np.array([r["a"]["diff"] for r in rows])
        N.update(Q_partitions=len(rows), Q_fedpub_min=fp.min(), Q_fedpub_max=fp.max(),
                 Q_fedpub_sd_of_means=fp.std(ddof=1), Q_fedavg_sd_of_means=fa.std(ddof=1),
                 Q_soft_sd_of_means=so.std(ddof=1), Q_deficit_min=-dd.max(), Q_deficit_max=-dd.min(),
                 Q_deficit_ratio=dd.min() / dd.max(), Q_fedpub_total_wins=int(sum(r["a"]["wins"] for r in rows)),
                 Q_fedpub_max_holm=max(r["a"]["holm"] for r in rows),
                 Q_rank_fedpub_last_everywhere=bool(all(r["fp"] < min(r["fa"], r["so"]) for r in rows)))
    return t


def table_sensitivity(df):
    w, we = W(df, "N"), W(df, "N", "ap_local_es")
    if FEDPUB_W not in w:
        return pd.DataFrame()
    rows = []
    for label, a, b in (("weighted minus official, full budget", w[FEDPUB_W], w[FEDPUB]),
                        ("weighted minus official, early-stopped checkpoint", we[FEDPUB_W], we[FEDPUB]),
                        ("weighted FED-PUB minus FedAvg, full budget", w[FEDPUB_W], w["FedAvg_GCN"]),
                        ("official FED-PUB minus FedAvg, full budget", w[FEDPUB], w["FedAvg_GCN"])):
        c = cmp(a, b)
        rows.append({"comparison": label, "n": c["n"], "mean a": f4(a.mean()), "mean b": f4(b.mean()),
                     "a minus b [95% CI]": ci(c), "wins a": c["wins"], "p": pf(c["p"])})
        N["sens_" + label.split(",")[0].replace(" ", "_") + ("_es" if "early" in label else "")] = c["diff"]
        N["sens_p_" + label.split(",")[0].replace(" ", "_") + ("_es" if "early" in label else "")] = c["p"]
    gap_off = cmp(w["FedAvg_GCN"], w[FEDPUB])["diff"]; gap_w = cmp(w["FedAvg_GCN"], w[FEDPUB_W])["diff"]
    N.update(sens_gap_official=gap_off, sens_gap_weighted=gap_w, sens_gap_share_explained=1 - gap_w / gap_off,
             sens_val_ap_official=df[(df.protocol == "N") & (df.variant == FEDPUB)].best_val_ap.mean(),
             sens_val_ap_weighted=df[(df.protocol == "N") & (df.variant == FEDPUB_W)].best_val_ap.mean(),
             N_val_ap_fedavg=df[(df.protocol == "N") & (df.variant == "FedAvg_GCN")].best_val_ap.mean())
    return pd.DataFrame(rows)


def table_sensitivity_lr_local(df):
    """PL (learning rate 1e-3) and PO (local training, no communication) against PN and FedAvg; protocol N"""
    w = W(df, "N")
    rows = []
    for key, label, va, vb in (("lr_minus_official", "lr 1e-3 minus official lr 0.01", FEDPUB_L, FEDPUB),
                               ("lr_minus_fedavg", "FED-PUB (lr 1e-3) minus FedAvg", FEDPUB_L, "FedAvg_GCN"),
                               ("official_minus_local", "FED-PUB minus Local (both lr 0.01)", FEDPUB, FEDPUB_O),
                               ("local_minus_fedavg", "Local minus FedAvg", FEDPUB_O, "FedAvg_GCN")):
        if va not in w or vb not in w:
            continue
        c = cmp(w[va], w[vb])
        rows.append({"comparison": label, "n": c["n"], "mean a": f4(w[va].mean()), "mean b": f4(w[vb].mean()),
                     "a minus b [95% CI]": ci(c), "wins a": c["wins"], "p": pf(c["p"])})
        N["sens2_" + key], N["sens2_p_" + key], N["sens2_wins_" + key] = c["diff"], c["p"], c["wins"]
    for v, key in ((FEDPUB_L, "lr"), (FEDPUB_O, "local")):
        sub = df[(df.protocol == "N") & (df.variant == v)]
        if len(sub):
            N[f"sens2_{key}_ap"], N[f"sens2_{key}_val_ap"] = sub.ap_local.mean(), sub.best_val_ap.mean()
            N[f"sens2_{key}_best_round_median"] = float(sub.best_round.median())
    return pd.DataFrame(rows)


def table_es(df):
    """FED-PUB: full-budget checkpoint against the checkpoint the patience-20 rule returns"""
    rows = []
    for proto in ["N"] + [f"Q{k}" for k in range(10)]:
        for v in FEDPUB_ALL:
            sub = df[(df.protocol == proto) & (df.variant == v)]
            if sub.empty:
                continue
            c = cmp(sub.set_index("seed").ap_local, sub.set_index("seed").ap_local_es)
            rows.append({"protocol": proto, "design": SHORT[v], "n": c["n"], "AP full": f4(sub.ap_local.mean()),
                         "AP early-stopped": f4(sub.ap_local_es.mean()), "full minus ES [95% CI]": ci(c),
                         "p": pf(c["p"]), "best round (median)": int(sub.best_round.median()),
                         "ES stop round (median)": int(sub.es_stop.median())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-results", default="results")
    ap.add_argument("--paper-results", required=True)
    ap.add_argument("--frozen-repo", required=True)
    ap.add_argument("--out", default="analysis/output")
    a = ap.parse_args()
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)

    df, prov = load(a.e1_results, a.paper_results, a.frozen_repo)
    counts = df.pivot_table(index="variant", columns="protocol", values="seed", aggfunc="count", fill_value=0)
    T = {"run_counts": counts.reset_index(), "partition_check": partition_check(df),
         "protocols": table_protocols(df), "n_all_designs": table_n_all(df), "fixed_partitions": table_fixed(df),
         "sensitivity_weighted_loss": table_sensitivity(df),
         "sensitivity_lr_local": table_sensitivity_lr_local(df), "early_stopping": table_es(df),
         "provenance": prov}
    fp_hashes = sorted(set(df[df.variant.isin(FEDPUB_ALL)].code_hash.dropna()))
    N["fedpub_code_hashes"] = fp_hashes
    N["n_fedpub_runs"] = int(df.variant.isin(FEDPUB_ALL).sum())
    df.drop(columns="fp").to_csv(out / "runs.csv", index=False)
    for k, t in T.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)

    titles = {"run_counts": "Completed runs per design and protocol",
              "partition_check": "Partition identity check: FED-PUB clients equal the paper's clients (same seed)",
              "protocols": "FED-PUB against FedAvg and Soft, per-client test AP (Holm over the two protocols)",
              "n_all_designs": "Protocol N, per-client test AP: every federated design against FedAvg (Holm over designs)",
              "fixed_partitions": "Fixed Louvain partitions Q0-Q9, full budget, per-client test AP (Holm over partitions)",
              "sensitivity_weighted_loss": "Sensitivity: class-weighted loss (PW) against the official loss (PN), protocol N",
              "sensitivity_lr_local": "Sensitivity: learning rate 1e-3 (PL) and local training without communication (PO), protocol N",
              "early_stopping": "FED-PUB: full-budget against early-stopped checkpoint",
              "provenance": "Provenance: code and configuration hashes per job"}
    md = ["# E1 FED-PUB analysis", "",
          f"FED-PUB runs: {N['n_fedpub_runs']}. Metric: per-client test AP. FED-PUB code hashes: "
          f"{', '.join(fp_hashes)} (later hashes only add the weighted-loss, learning-rate and local-training options; the official path is unchanged).", ""]
    for k, t in T.items():
        md += [f"## {titles[k]}", "", _md(t) if len(t) else "(no data)", ""]
    md += ["## Key numbers", ""] + [f"- {k}: {v if isinstance(v, (list, bool, str)) else round(float(v), 4)}"
                                    for k, v in N.items()]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "numbers.json").write_text(json.dumps(
        {k: (v if isinstance(v, (list, bool, str)) else float(v)) for k, v in N.items()}, indent=2), encoding="utf-8")
    print("\n".join(md[:3]))
    print(f"written: {out / 'report.md'}")
    if not N.get("partition_check_all_identical", False):
        print("WARNING: partition check failed or incomplete, see tables/partition_check.csv")


if __name__ == "__main__":
    main()
