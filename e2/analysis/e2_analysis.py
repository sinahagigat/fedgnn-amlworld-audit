"""E2 (AMLworld) analysis: replicates the statistics of paper 1 on dataset 2

    python analysis/e2_analysis.py [--results results] [--out analysis/output]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FED6 = ["FedAvg_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN",
        "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"]
SHORT = {"Centralized_GCN": "Central-GCN", "Centralized_MLP": "Central-MLP", "FedAvg_GCN": "FedAvg",
         "FedProto_GCN": "FedProto", "FedGNN_Hard_GCN": "Hard", "FedGNN_Soft_GCN": "Soft",
         "FedGNN_Soft_Virtual_GCN": "Soft-VN", "FedGNN_Soft_Virtual_GAT": "Soft-VN-GAT",
         "FedAvg_MLP": "FedAvg-MLP"}
PARTS = [("AB", "bank"), ("AL", "Louvain"), ("AC", "country"), ("AR", "random")]
N = {}


# loading
def load(results):
    rows = []
    for d in sorted(Path(results).iterdir()):
        f = d / "artifacts.json"
        if not f.is_file() or d.name[:2] not in ("AB", "AL", "AC", "AR", "NB", "NC", "F0", "F1", "F2", "F3", "F4"):
            continue
        for r in json.loads(f.read_text(encoding="utf-8"))["runs"].values():
            if r.get("status") != "completed":
                continue
            tm = r["test_metrics"]
            es = r.get("test_metrics_es") or {}
            pdg = r.get("partition_diagnostics") or {}
            rows.append(dict(
                exp=d.name[:2], job=d.name, variant=r["variant_name"], seed=int(r["seed"]),
                ap=tm["ap"], auc=tm["auc"],
                ap_local=(r.get("test_metrics_local") or {}).get("ap", np.nan),
                ap_es=es.get("ap", np.nan), es_stop=r.get("es_stop_round", np.nan),
                es_best=r.get("es_best_round_idx", np.nan),
                best_round=r.get("best_round_idx", np.nan), rounds=r.get("total_rounds_run", np.nan),
                retention=pdg.get("edge_retention_rate", np.nan), runtime_h=r.get("runtime_sec", np.nan) / 3600))
    df = pd.DataFrame(rows)
    dup = df.duplicated(["exp", "variant", "seed"])
    assert not dup.any(), f"duplicate runs: {df[dup][['exp', 'variant', 'seed']].values.tolist()}"
    return df


def wide(df, exp, col="ap"):
    return df[df.exp == exp].pivot(index="seed", columns="variant", values=col)


# ---------------------------------------------------------------- statistics
def pair(a, b):
    """align two seed-indexed series on their common seeds"""
    s = a.dropna().index.intersection(b.dropna().index)
    return a.loc[s].values, b.loc[s].values


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
    """paired comparison a - b: dict with n, mean, CI, Wilcoxon p, wins"""
    x, y = pair(a, b); d = x - y; lo, hi = ci95(d)
    return dict(n=len(d), diff=d.mean() if len(d) else np.nan, lo=lo, hi=hi, p=wil(x, y), wins=int((d > 0).sum()))


def null_range(W, cols, reps=20000, seed=0):
    X = W[cols].dropna().values
    obs = X.mean(0).max() - X.mean(0).min()
    rng = np.random.default_rng(seed); null = np.empty(reps)
    for i in range(reps):
        P = np.array([row[rng.permutation(len(row))] for row in X]); m = P.mean(0); null[i] = m.max() - m.min()
    return obs, float((null >= obs).mean()), X.shape[0]


def _md(t):
    try:
        return t.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return "```\n" + t.to_string(index=False) + "\n```"


def f4(x): return "" if pd.isna(x) else f"{x:.4f}"
def s4(x): return "" if pd.isna(x) else f"{x:+.4f}"
def pf(p): return "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")
def ci(c): return f"{s4(c['diff'])} [{s4(c['lo'])}, {s4(c['hi'])}]"


# analyses
def design_table(df, exp, central_from=None):
    """designs vs FedAvg-GCN, Holm over all compared designs"""
    W = wide(df, exp); WL = wide(df, exp, "ap_local")
    if central_from is not None and central_from != exp:
        WC = wide(df, central_from)
        for c in ("Centralized_GCN", "Centralized_MLP"):
            if c in WC:
                W[c] = WC[c]
    order = [v for v in ["Centralized_GCN"] + FED6 + ["FedAvg_MLP", "Centralized_MLP"] if v in W]
    others = [v for v in order if v != "FedAvg_GCN"]
    cs = {v: cmp(W[v], W["FedAvg_GCN"]) for v in others}
    ph = dict(zip(others, holm([cs[v]["p"] for v in others])))
    rows = []
    for v in order:
        c = cs.get(v)
        rows.append({"design": SHORT[v], "n": int(W[v].notna().sum()), "AP mean": f4(W[v].mean()),
                     "AP SD": f4(W[v].std(ddof=1)), "local AP": f4(WL[v].mean()) if v in WL else "",
                     "minus FedAvg [95% CI]": ci(c) if c else "", "wins": c["wins"] if c else "",
                     "Holm p": pf(ph[v]) if c else ""})
        N[f"{exp}_{SHORT[v]}_ap"] = W[v].mean()
        if c:
            N[f"{exp}_{SHORT[v]}_vs_fedavg"] = c["diff"]; N[f"{exp}_{SHORT[v]}_vs_fedavg_pholm"] = ph[v]
    return pd.DataFrame(rows), W


def pairwise_fed(W, exp):
    cols = [v for v in FED6 if v in W]
    pr = [(a, b, cmp(W[a], W[b])) for a, b in itertools.combinations(cols, 2)]
    ph = holm([c["p"] for _, _, c in pr])
    rows = [{"a": SHORT[a], "b": SHORT[b], "a minus b [95% CI]": ci(c), "wins a": c["wins"], "p": pf(c["p"]),
             "Holm p": pf(h)} for (a, b, c), h in zip(pr, ph)]
    N[f"{exp}_pairs_n"] = len(pr); N[f"{exp}_pairs_sig_holm"] = int((ph < 0.05).sum())
    obs, p, n = null_range(W, cols)
    N[f"{exp}_null_spread_obs"] = obs; N[f"{exp}_null_spread_p"] = p; N[f"{exp}_null_spread_n"] = n
    conv = [v for v in cols if v != "FedGNN_Soft_Virtual_GAT"]  # convolutional designs only, as in paper 1
    obs, p, n = null_range(W, conv)
    N[f"{exp}_null_spread_conv_obs"] = obs; N[f"{exp}_null_spread_conv_p"] = p
    return pd.DataFrame(rows)


def partition_table(df):
    WC = wide(df, "AB"); rows, raw = [], []
    for exp, name in PARTS:
        W = wide(df, exp)
        if "FedAvg_GCN" not in W:
            continue
        soft = cmp(W["FedGNN_Soft_GCN"], W["FedAvg_GCN"]); raw.append(soft["p"])
        graph = cmp(W["FedAvg_GCN"], W["FedAvg_MLP"])
        gap = cmp(WC["Centralized_GCN"], W["FedAvg_GCN"])
        ret = df[(df.exp == exp) & (df.variant == "FedAvg_GCN")].retention.mean()
        rows.append({"partition": name, "n": soft["n"], "edges kept": f"{100 * ret:.1f}%",
                     "FedAvg AP": f4(W.FedAvg_GCN.mean()), "Soft AP": f4(W.FedGNN_Soft_GCN.mean()),
                     "Soft minus FedAvg [95% CI]": ci(soft), "Soft wins": soft["wins"], "p": pf(soft["p"]),
                     "FedAvg GCN minus MLP [95% CI]": ci(graph), "GCN-MLP p": pf(graph["p"]),
                     "Central-GCN minus FedAvg [95% CI]": ci(gap), "gap p": pf(gap["p"])})
        N[f"part_{name}_retention"] = ret; N[f"part_{name}_soft_minus_fedavg"] = soft["diff"]
        N[f"part_{name}_graph_value_fed"] = graph["diff"]; N[f"part_{name}_gap"] = gap["diff"]
    t = pd.DataFrame(rows); t.insert(8, "Holm p (4 partitions)", [pf(x) for x in holm(raw)])
    return t


def full_budget(df):
    W, WE = wide(df, "NB"), wide(df, "NB", "ap_es")
    WC, WCE = wide(df, "NC"), wide(df, "NC", "ap_es")
    sub = df[df.exp.isin(["NB", "NC"])]
    rows = []
    for v in [c for c in ["Centralized_GCN", "Centralized_MLP"] if c in WC] + [c for c in FED6 + ["FedAvg_MLP"] if c in W]:
        src, srcE = (WC, WCE) if v.startswith("Centralized") else (W, WE)
        s = sub[sub.variant == v]
        e = cmp(src[v], srcE[v]) if not v.startswith("Centralized") else None
        rows.append({"design": SHORT[v], "n": int(src[v].notna().sum()),
                     "AP full budget": f4(src[v].mean()), "SD": f4(src[v].std(ddof=1)),
                     "AP early stopping (patience 20)": "" if v.startswith("Centralized") else f4(srcE[v].mean()),
                     "full minus ES [95% CI]": ci(e) if e else "", "p": pf(e["p"]) if e else "",
                     "ES fired": "" if v.startswith("Centralized") else f"{int((s.es_stop > 0).sum())}/{len(s)}",
                     "best round (full)": f"{s.best_round.mean():.0f}",
                     "hours/run": f"{s.runtime_h.mean():.1f}"})
        N[f"full_{SHORT[v]}_ap"] = src[v].mean()
        if e:
            N[f"full_{SHORT[v]}_ap_es"] = srcE[v].mean(); N[f"full_{SHORT[v]}_es_effect"] = e["diff"]
    t1 = pd.DataFrame(rows)

    fed = [v for v in FED6 + ["FedAvg_MLP"] if v in W and v != "FedAvg_GCN"]
    r2 = []
    for lab, X in (("full budget", W), ("early stopping", WE)):
        cs = [cmp(X[v], X["FedAvg_GCN"]) for v in fed]; ph = holm([c["p"] for c in cs])
        for v, c, h in zip(fed, cs, ph):
            r2.append({"protocol": lab, "design": SHORT[v], "n": c["n"], "minus FedAvg [95% CI]": ci(c),
                       "wins": c["wins"], "Holm p": pf(h)})
            N[f"full_{lab.split()[0]}_{SHORT[v]}_vs_fedavg_pholm"] = h
    t2 = pd.DataFrame(r2)

    # federated-centralized gap as in paper 1: centralized always trains the full budget
    r3 = []
    for enc, cen, fed_ in (("GCN", "Centralized_GCN", "FedAvg_GCN"), ("MLP (control)", "Centralized_MLP", "FedAvg_MLP")):
        if cen not in WC or fed_ not in W:
            continue
        g_es, g_full = cmp(WC[cen], WE[fed_]), cmp(WC[cen], W[fed_])
        r3.append({"encoder": enc, "n": g_full["n"], "Centralized AP": f4(WC[cen].mean()),
                   "FedAvg AP, ES": f4(WE[fed_].mean()), "gap, ES [95% CI]": ci(g_es), "p ES": pf(g_es["p"]),
                   "FedAvg AP, full": f4(W[fed_].mean()), "gap, full [95% CI]": ci(g_full), "p full": pf(g_full["p"])})
        k = "gcn" if enc == "GCN" else "mlp"
        N[f"gap_{k}_es"] = g_es["diff"]; N[f"gap_{k}_full"] = g_full["diff"]
        N[f"gap_{k}_share_from_es"] = (g_es["diff"] - g_full["diff"]) / g_es["diff"] if g_es["diff"] else np.nan
    t3 = pd.DataFrame(r3)

    r4 = []
    for lab, a, b in (("centralized", WC.get("Centralized_GCN"), WC.get("Centralized_MLP")),
                      ("federated (FedAvg), full", W.get("FedAvg_GCN"), W.get("FedAvg_MLP")),
                      ("federated (FedAvg), ES", WE.get("FedAvg_GCN"), WE.get("FedAvg_MLP"))):
        if a is None or b is None:
            continue
        c = cmp(a, b)
        r4.append({"setting": lab, "n": c["n"], "GCN minus MLP [95% CI]": ci(c), "wins": c["wins"], "p": pf(c["p"])})
        N[f"graph_value_{lab.split()[0]}{'_es' if 'ES' in lab else ''}"] = c["diff"]
    if "graph_value_centralized" in N and "graph_value_federated" in N:
        N["graph_value_lost_share"] = 1 - N["graph_value_federated"] / N["graph_value_centralized"]
    return t1, t2, t3, pd.DataFrame(r4)


def budget_effect(df):
    B1, B2 = wide(df, "AB"), wide(df, "NB")
    C2 = wide(df, "NC")
    for c in ("Centralized_GCN", "Centralized_MLP"):
        if c in C2:
            B2[c] = C2[c]
    rows = []
    for v in [x for x in ["Centralized_GCN"] + FED6 + ["FedAvg_MLP", "Centralized_MLP"] if x in B1 and x in B2]:
        c = cmp(B2[v], B1[v])
        rows.append({"design": SHORT[v], "n": c["n"], "budget 1 AP": f4(B1[v].mean()), "full budget AP": f4(B2[v].mean()),
                     "full minus budget 1 [95% CI]": ci(c), "p": pf(c["p"])})
    return pd.DataFrame(rows)


def vs_fedavg(W, cols, tag):
    """federated designs against FedAvg-GCN on one wide table, Holm over the designs"""
    cols = [v for v in cols if v in W and v != "FedAvg_GCN"]
    cs = [cmp(W[v], W["FedAvg_GCN"]) for v in cols]; ph = holm([c["p"] for c in cs])
    rows = []
    for v, c, h in zip(cols, cs, ph):
        rows.append({"design": SHORT[v], "n": c["n"], "AP": f4(W[v].mean()), "minus FedAvg [95% CI]": ci(c),
                     "wins": c["wins"], "Holm p": pf(h)})
        N[f"{tag}_{SHORT[v]}_vs_fedavg"] = c["diff"]; N[f"{tag}_{SHORT[v]}_vs_fedavg_pholm"] = h
    return rows


def per_client(df):
    """same comparisons under per-client inference (each client scores its own nodes on its own subgraph)"""
    out = {}
    # budget 1: designs against FedAvg and pairwise counts
    for exp, name in (("AB", "bank"), ("AL", "Louvain")):
        if not (df.exp == exp).any():
            continue
        L = wide(df, exp, "ap_local")
        rows = [{"design": "FedAvg", "n": int(L.FedAvg_GCN.notna().sum()), "AP": f4(L.FedAvg_GCN.mean()),
                 "minus FedAvg [95% CI]": "", "wins": "", "Holm p": ""}]
        rows += vs_fedavg(L, FED6 + ["FedAvg_MLP"], f"{exp}_local")
        out[f"local_b1_{name}_designs"] = pd.DataFrame(rows)
        cols = [v for v in FED6 if v in L]
        pr = [cmp(L[a], L[b]) for a, b in itertools.combinations(cols, 2)]
        N[f"{exp}_local_pairs_sig_holm"] = int((holm([c["p"] for c in pr]) < 0.05).sum()); N[f"{exp}_local_pairs_n"] = len(pr)
    WC = wide(df, "AB"); rows, raw = [], []
    for exp, name in PARTS:
        L = wide(df, exp, "ap_local")
        if "FedAvg_GCN" not in L:
            continue
        s = cmp(L.FedGNN_Soft_GCN, L.FedAvg_GCN); raw.append(s["p"])
        g = cmp(L.FedAvg_GCN, L.FedAvg_MLP); gap = cmp(WC.Centralized_GCN, L.FedAvg_GCN)
        rows.append({"partition": name, "n": s["n"], "FedAvg AP": f4(L.FedAvg_GCN.mean()), "Soft AP": f4(L.FedGNN_Soft_GCN.mean()),
                     "Soft minus FedAvg [95% CI]": ci(s), "Soft wins": s["wins"], "p": pf(s["p"]),
                     "FedAvg GCN minus MLP [95% CI]": ci(g), "GCN-MLP p": pf(g["p"]),
                     "Central-GCN minus FedAvg [95% CI]": ci(gap), "gap p": pf(gap["p"])})
        N[f"local_part_{name}_soft_minus_fedavg"] = s["diff"]; N[f"local_part_{name}_graph_value_fed"] = g["diff"]
        N[f"local_part_{name}_gap"] = gap["diff"]
    if rows:
        t = pd.DataFrame(rows); t.insert(7, "Holm p (4 partitions)", [pf(x) for x in holm(raw)])
        for r, h in zip(rows, holm(raw)):
            N[f"local_part_{r['partition']}_soft_pholm"] = h
        out["local_b1_partitions"] = t
    if (df.exp == "NB").any() and (df.exp == "NC").any():
        L, G, C = wide(df, "NB", "ap_local"), wide(df, "NB"), wide(df, "NC")
        rows = [{"design": "FedAvg", "n": int(L.FedAvg_GCN.notna().sum()), "AP": f4(L.FedAvg_GCN.mean()),
                 "minus FedAvg [95% CI]": "", "wins": "", "Holm p": ""}]
        rows += vs_fedavg(L, FED6 + ["FedAvg_MLP"], "full_local")
        out["local_full_vs_fedavg"] = pd.DataFrame(rows)
        r3 = []
        for enc, cen, fed in (("GCN", "Centralized_GCN", "FedAvg_GCN"), ("MLP (control)", "Centralized_MLP", "FedAvg_MLP")):
            gg, gl = cmp(C[cen], G[fed]), cmp(C[cen], L[fed])
            r3.append({"encoder": enc, "n": gl["n"], "Centralized AP": f4(C[cen].mean()),
                       "FedAvg AP, global": f4(G[fed].mean()), "gap, global [95% CI]": ci(gg), "p global": pf(gg["p"]),
                       "FedAvg AP, per-client": f4(L[fed].mean()), "gap, per-client [95% CI]": ci(gl), "p per-client": pf(gl["p"])})
            N[f"gap_{'gcn' if enc == 'GCN' else 'mlp'}_full_local"] = gl["diff"]
        out["local_full_gap"] = pd.DataFrame(r3)
        g = cmp(L.FedAvg_GCN, L.FedAvg_MLP)
        out["local_graph_value"] = pd.DataFrame([{"setting": "federated (FedAvg), full, per-client", "n": g["n"],
                                                  "GCN minus MLP [95% CI]": ci(g), "wins": g["wins"], "p": pf(g["p"])}])
        N["graph_value_federated_local"] = g["diff"]; N["graph_value_federated_local_p"] = g["p"]
    return out


def references(df, path):
    if not Path(path).is_file():
        return None
    R = pd.read_csv(path)
    WC, WG, WL = wide(df, "NC"), wide(df, "NB"), wide(df, "NB", "ap_local")
    rows = []
    for m, g in R.groupby("model", sort=False):
        ap = g.test_ap
        rows.append({"model": m, "fits": len(g), "test AP": f4(ap.mean()), "SD": f4(ap.std(ddof=1)) if len(g) > 1 else "",
                     "test ROC-AUC": f4(g.test_auc.mean()),
                     "minus Central-GCN": s4(ap.mean() - WC.Centralized_GCN.mean()),
                     "minus FedAvg (global)": s4(ap.mean() - WG.FedAvg_GCN.mean()),
                     "minus FedAvg (per-client)": s4(ap.mean() - WL.FedAvg_GCN.mean())})
        k = m.split()[0].lower()
        N[f"ref_{k}_ap"] = ap.mean(); N[f"ref_{k}_minus_central_gcn"] = ap.mean() - WC.Centralized_GCN.mean()
    return pd.DataFrame(rows)


def fixed_partitions(df):
    """five fixed Louvain partitions (F0-F4), budget 1: Soft minus FedAvg on each, Holm over partitions"""
    exps = [e for e in ("F0", "F1", "F2", "F3", "F4") if (df.exp == e).any()]
    if not exps:
        return None
    rows = []
    for col, lab in (("ap", "global"), ("ap_local", "per-client")):
        cs = [cmp(wide(df, e, col).FedGNN_Soft_GCN, wide(df, e, col).FedAvg_GCN) for e in exps]
        ph = holm([c["p"] for c in cs])
        for e, c, h in zip(exps, cs, ph):
            ret = df[(df.exp == e) & (df.variant == "FedAvg_GCN")].retention.mean()
            rows.append({"inference": lab, "partition": e, "n": c["n"], "edges kept": f"{100 * ret:.2f}%",
                         "FedAvg AP": f4(wide(df, e, col).FedAvg_GCN.mean()),
                         "Soft minus FedAvg [95% CI]": ci(c), "Soft wins": c["wins"], "p": pf(c["p"]),
                         "Holm p (partitions)": pf(h)})
            N[f"fixed_{e}_{lab}_soft_minus_fedavg"] = c["diff"]; N[f"fixed_{e}_{lab}_pholm"] = h
        d = np.array([c["diff"] for c in cs])
        N[f"fixed_{lab}_n_better"] = int(((ph < 0.05) & (d > 0)).sum())
        N[f"fixed_{lab}_n_worse"] = int(((ph < 0.05) & (d < 0)).sum())
        N[f"fixed_{lab}_diff_min"] = float(d.min()); N[f"fixed_{lab}_diff_max"] = float(d.max())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here.parent / "results"))
    ap.add_argument("--out", default=str(here / "output"))
    a = ap.parse_args()
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)

    df = load(a.results)
    df.to_csv(out / "runs.csv", index=False)
    cnt = df.groupby(["exp", "variant"]).size().unstack(0).fillna(0).astype(int)
    T = {"run_counts": cnt.reset_index()}

    T["b1_bank_designs"], WB = design_table(df, "AB")
    T["b1_bank_pairwise"] = pairwise_fed(WB, "AB")
    if (df.exp == "AL").any():
        T["b1_louvain_designs"], WL = design_table(df, "AL", central_from="AB")
        T["b1_louvain_pairwise"] = pairwise_fed(WL, "AL")
    T["b1_partitions"] = partition_table(df)
    if (df.exp == "NB").any() and (df.exp == "NC").any():
        T["full_designs"], T["full_vs_fedavg"], T["full_gap"], T["graph_value"] = full_budget(df)
        T["budget_effect"] = budget_effect(df)
    T.update(per_client(df))
    fx = fixed_partitions(df)
    if fx is not None:
        T["fixed_partitions"] = fx
    ref = references(df, out / "references.csv")
    if ref is not None:
        T["references"] = ref

    for k, t in T.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    clean = {k: (None if pd.isna(v) else float(v)) for k, v in N.items()}
    (out / "numbers.json").write_text(json.dumps(clean, indent=1))

    title = {"run_counts": "Completed runs per experiment and design",
             "b1_bank_designs": "Budget 1, bank partition: designs against FedAvg (Holm over designs)",
             "b1_bank_pairwise": "Budget 1, bank partition: all pairs of the six federated GCN designs",
             "b1_louvain_designs": "Budget 1, Louvain partition: designs against FedAvg (central runs from AB, same seeds)",
             "b1_louvain_pairwise": "Budget 1, Louvain partition: all pairs of the six federated GCN designs",
             "b1_partitions": "Budget 1: partition schemes (Central-GCN from AB, same seeds)",
             "full_designs": "Full budget (300 rounds x 5 local steps): with and without early stopping",
             "full_vs_fedavg": "Full budget: federated designs against FedAvg (Holm within protocol)",
             "full_gap": "Federated-centralized gap (centralized always trains the full budget, as in paper 1)",
             "graph_value": "Value of the graph: GCN minus MLP",
             "budget_effect": "Budget effect, bank partition: full budget minus budget 1",
             "local_b1_bank_designs": "PER-CLIENT inference. Budget 1, bank partition: designs against FedAvg",
             "local_b1_Louvain_designs": "PER-CLIENT inference. Budget 1, Louvain partition: designs against FedAvg",
             "local_b1_partitions": "PER-CLIENT inference. Budget 1: partition schemes",
             "local_full_vs_fedavg": "PER-CLIENT inference. Full budget: federated designs against FedAvg (Holm)",
             "local_full_gap": "Federated-centralized gap, full budget: global vs per-client inference",
             "local_graph_value": "PER-CLIENT inference. Value of the graph in FedAvg, full budget",
             "fixed_partitions": "Budget 1, five FIXED Louvain partitions (F0-F4, fixed_partition_seed 0-4): Soft minus FedAvg",
             "references": "Non-graph references on the same split (analysis/aml_references.py; RF over seeds, HGB and LR deterministic)"}
    md = ["# E2 AMLworld analysis", "",
          f"Completed runs: {len(df)}. Seeds per comparison are the common seeds of both arms (column n).", ""]
    for k, t in T.items():
        md += [f"## {title.get(k, k)}", "", _md(t), ""]
    md += ["## Key numbers", ""]
    for k in [x for x in ("AB_pairs_sig_holm", "AB_pairs_n", "AB_null_spread_obs", "AB_null_spread_p", "AB_null_spread_conv_obs", "AB_null_spread_conv_p",
                          "AL_pairs_sig_holm", "AL_null_spread_p", "AL_null_spread_conv_p", "gap_gcn_es", "gap_gcn_full", "gap_gcn_share_from_es",
                          "graph_value_centralized", "graph_value_federated", "graph_value_lost_share", "gap_gcn_full_local", "graph_value_federated_local", "AB_local_pairs_sig_holm", "AL_local_pairs_sig_holm") if x in clean]:
        md.append(f"- {k}: {clean[k]:.4f}" if isinstance(clean[k], float) else f"- {k}: {clean[k]}")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done: {len(df)} runs -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
