"""stopping-rule control and per-client mean AP on Elliptic (Tables S35 and S36)

    python analysis/e3_analysis.py --e1-results results
        --paper-results <fedgnn-elliptic-audit results>
        --frozen-repo <fedgnn-elliptic-audit>
        --data-dir <elliptic data>
        [--out analysis/output_e3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
N = {}


# ------------------------------------------------------------------ statistics
def wil(x, y):
    d = np.asarray(x) - np.asarray(y)
    if len(d) < 2 or np.all(d == 0):
        return 1.0
    method = "exact" if len(d) <= 25 and np.all(d != 0) else "auto"
    return float(stats.wilcoxon(x, y, method=method).pvalue)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); out = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); out[k] = min(run, 1.0)
    return out


def cmp(a, b):
    s = a.dropna().index.intersection(b.dropna().index)
    x, y = a.loc[s].values, b.loc[s].values; d = x - y; n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, diff=d.mean(), lo=d.mean() - h, hi=d.mean() + h, p=wil(x, y), wins=int((d > 0).sum()))


def fmt(c):
    return f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def pf(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _md(t):
    try:
        return t.to_markdown(index=False)
    except ImportError:
        return "```\n" + t.to_string(index=False) + "\n```"


# ------------------------------------------------------------------ run records
def runs_of(dirs):
    rows = {}
    for d in dirs:
        f = d / "artifacts.json"
        if not f.is_file():
            f = d / "artifacts_slim.json"
        if not f.is_file():
            continue
        for r in json.loads(f.read_text(encoding="utf-8"))["runs"].values():
            if r.get("status") != "completed":
                continue
            tm, te = r.get("test_metrics") or {}, r.get("test_metrics_es") or {}
            rows[(r["variant_name"], int(r["seed"]))] = dict(
                job=d.name, ap=tm.get("ap", np.nan), ap_es=te.get("ap", np.nan),
                es_stop=r.get("es_stop_round", np.nan), best=r.get("best_round_idx", np.nan),
                ap_local=(r.get("test_metrics_local") or {}).get("ap", np.nan),
                pc=(r.get("partition_diagnostics") or {}).get("per_client"),
                pred=(d / "predictions" / r["predictions_file"]) if r.get("predictions_file") else None)
    return rows


def series(rows, variant, key):
    s = {seed: v[key] for (var, seed), v in rows.items() if var == variant}
    return pd.Series(s, dtype=float).sort_index()


def jobs(root, prefix, n):
    return [Path(root) / f"{prefix}{i}" for i in range(1, n + 1)]


# ------------------------------------------------------------------ part 1
def part1(a):
    ce = runs_of(jobs(a.e1_results, "CE", 4))
    ch = runs_of(jobs(a.e1_results, "CH", 4))
    fn = runs_of(jobs(a.e1_results, "FN", 4))
    d = runs_of(jobs(a.paper_results, "D", 10))
    e = runs_of(jobs(a.paper_results, "E", 4))
    n = runs_of(sorted(Path(a.frozen_repo).glob("*/results/v7/N[1-4]")))

    checks, rows = [], []
    for enc, cen, fed, ref in (("GCN", "Centralized_GCN", "FedAvg_GCN", d), ("MLP", "Centralized_MLP", "FedAvg_MLP", e)):
        c_full, c_es20 = series(ce, cen, "ap"), series(ce, cen, "ap_es")
        c_es100 = series(ch, cen, "ap_es")
        f_full, f_es = series(fn, fed, "ap"), series(fn, fed, "ap_es")
        for label, mine, paper in ((f"{enc}: centralized full (CE) = paper budget 2", c_full, series(ref, cen, "ap")),
                                   (f"{enc}: centralized full (CH) = CE", series(ch, cen, "ap"), c_full),
                                   (f"{enc}: FedAvg full (FN) = paper N", f_full, series(n, fed, "ap")),
                                   (f"{enc}: FedAvg patience-20 checkpoint (FN) = paper budget 2", f_es, series(ref, fed, "ap"))):
            s = mine.index.intersection(paper.index)
            dev = float(np.max(np.abs(mine.loc[s] - paper.loc[s]))) if len(s) else np.nan
            checks.append({"check": label, "seeds": len(s), "max abs difference": f"{dev:.2e}"})
            N[f"check_{enc}_{len(checks)}"] = dev
        fires = series(ce, cen, "es_stop"); fires100 = series(ch, cen, "es_stop")
        for rule, c, f in (("paper: federated stops (patience 20 rounds), centralized does not", c_full, f_es),
                           ("both stop, patience 20 (rounds / steps)", c_es20, f_es),
                           ("both stop, patience 20 rounds / 100 steps (same local steps)", c_es100, f_es),
                           ("neither stops (full budget)", c_full, f_full)):
            cc = cmp(c, f)
            rows.append({"encoder": enc, "rule": rule, "n": cc["n"], "centralized AP": f"{c.mean():.4f}",
                         "FedAvg AP": f"{f.mean():.4f}", "gap (centralized - FedAvg) [95% CI]": fmt(cc),
                         "centralized ahead": f"{cc['wins']}/{cc['n']}", "p": pf(cc["p"])})
            N[f"gap_{enc}_{rule.split(':')[0].split(',')[0].replace(' ', '_')}"] = cc["diff"]
        rows.append({"encoder": enc, "rule": "centralized: rule fires (patience 20 / 100 steps), median stop step",
                     "n": int(fires.notna().sum()),
                     "centralized AP": f"{int((fires >= 0).sum())} / {int((fires100 >= 0).sum())} runs",
                     "FedAvg AP": "", "gap (centralized - FedAvg) [95% CI]":
                         f"median {np.nanmedian(fires[fires >= 0]) if (fires >= 0).any() else np.nan:.0f} / "
                         f"{np.nanmedian(fires100[fires100 >= 0]) if (fires100 >= 0).any() else np.nan:.0f}",
                     "centralized ahead": "", "p": ""})
    return pd.DataFrame(checks), pd.DataFrame(rows)


# part 2
def load_graph(data_dir):
    from src.compat import install_config_compat
    import src.config as cfgmod
    install_config_compat(cfgmod)
    from src.data.elliptic import load_elliptic_raw, preprocess_elliptic
    cfg = cfgmod.load_config(str(ROOT / "configs" / "e1_PN_louvain_full.yaml"))
    cfg.data.raw_dir = str(Path(data_dir)); cfg.data.cache_dir = str(Path(data_dir) / "cache")
    g = preprocess_elliptic(load_elliptic_raw(cfg.data), cfg.data)
    g = g[0] if isinstance(g, (tuple, list)) else g
    return cfg, g


def assignment_for(cfg, g, seed, cache):
    if seed not in cache:
        from src.data.partition import partition_graph_federated
        from src.utils.common import set_global_seed
        set_global_seed(seed, deterministic=True)
        assign, diag = partition_graph_federated(g, cfg.partition, seed)
        cache[seed] = (np.asarray(assign), diag)
    return cache[seed]


def fingerprint(pc):
    return tuple((int(v["num_nodes"]), int(v["num_train"]), int(v["num_intra_edges"]))
                 for _, v in sorted(pc.items(), key=lambda kv: int(kv[0])))


def client_aps(y, p, cl):
    aps, npos = [], []
    for c in np.unique(cl):
        m = cl == c
        if 0 < y[m].sum() < m.sum():
            aps.append(average_precision_score(y[m], p[m])); npos.append(int(y[m].sum()))
    aps, npos = np.array(aps), np.array(npos)
    return aps.mean(), float((aps * npos).sum() / npos.sum()), len(aps)


def part2(a):
    cfg, g = load_graph(a.data_dir)
    tm = g.test_mask.cpu().numpy().astype(bool)
    y_all = g.y.cpu().numpy()[tm].astype(int)
    cache, rows, bad = {}, [], []
    sources = {"A": [(runs_of(jobs(a.paper_results, "A", 4)), ("FedAvg_GCN", "FedGNN_Soft_GCN")),
                     (runs_of(jobs(a.e1_results, "PA", 4)), None)],
               "N": [(runs_of(jobs(a.e1_results, "FN", 4)), ("FedAvg_GCN",)),
                     (runs_of([Path(a.e1_results) / f"{p}{i}" for p in ("PN", "PW", "PL", "PO") for i in range(1, 5)]), None)]}
    for proto, srcs in sources.items():
        for rr, keep in srcs:
            for (var, seed), r in rr.items():
                if keep is not None and var not in keep:
                    continue
                if r["pred"] is None or not r["pred"].is_file():
                    bad.append(f"{proto} {var} {seed}: no prediction file"); continue
                z = np.load(r["pred"])
                if "p_test_local" not in z.files:
                    bad.append(f"{proto} {var} {seed}: no per-client scores"); continue
                assign, diag = assignment_for(cfg, g, seed, cache)
                if r["pc"] is not None and fingerprint(diag["per_client"]) != fingerprint(r["pc"]):
                    bad.append(f"{proto} {var} {seed}: rebuilt partition differs from the run record"); continue
                y, p = z["y_test"].astype(int), z["p_test_local"].astype(np.float64)
                if not np.array_equal(y, y_all):
                    bad.append(f"{proto} {var} {seed}: test labels out of order"); continue
                pooled = average_precision_score(y, p)
                if abs(pooled - r["ap_local"]) > 1e-6:
                    bad.append(f"{proto} {var} {seed}: pooled AP {pooled:.6f} != record {r['ap_local']:.6f}"); continue
                mean_ap, wmean_ap, k = client_aps(y, p, assign[tm])
                rows.append(dict(protocol=proto, variant=var, seed=seed, pooled=pooled, client_mean=mean_ap,
                                 client_wmean=wmean_ap, clients_scored=k))
    df = pd.DataFrame(rows, columns=["protocol", "variant", "seed", "pooled", "client_mean",
                                      "client_wmean", "clients_scored"])
    out = []
    for proto in ("A", "N"):
        sub = df[df.protocol == proto]
        fedpub = sorted(v for v in sub.variant.unique() if v.startswith("FedPub"))
        pairs = [(v, "FedAvg_GCN") for v in fedpub]
        if proto == "A":
            pairs += [(v, "FedGNN_Soft_GCN") for v in fedpub] + [("FedGNN_Soft_GCN", "FedAvg_GCN")]
        for metric in ("pooled", "client_mean", "client_wmean"):
            w = sub.pivot(index="seed", columns="variant", values=metric)
            pairs_ok = [(v, ref) for v, ref in pairs if v in w and ref in w]
            cs = [cmp(w[v], w[ref]) for v, ref in pairs_ok]
            for (v, ref), c, h in zip(pairs_ok, cs, holm([c["p"] for c in cs]) if cs else []):
                out.append({"protocol": proto, "metric": metric, "design": v, "vs": ref, "n": c["n"],
                            "design AP": f"{w[v].mean():.4f}", "reference AP": f"{w[ref].mean():.4f}",
                            "difference [95% CI]": fmt(c), "design ahead": f"{c['wins']}/{c['n']}", "Holm p": pf(h)})
                N[f"pc_{proto}_{metric}_{v}_vs_{ref}"] = c["diff"]
    N["pc_problems"] = len(bad)
    N["pc_runs"] = len(df)
    N["pc_clients_scored_min"] = int(df.clients_scored.min()) if len(df) else 0
    return df, pd.DataFrame(out), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-results", default="results")
    ap.add_argument("--paper-results", required=True)
    ap.add_argument("--frozen-repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="analysis/output_e3")
    a = ap.parse_args()
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)

    checks, gaps = part1(a)
    runs, pc, bad = part2(a)
    checks.to_csv(out / "tables" / "reproduction_checks.csv", index=False)
    gaps.to_csv(out / "tables" / "gap_symmetric_stopping.csv", index=False)
    pc.to_csv(out / "tables" / "per_client_mean_ap.csv", index=False)
    runs.to_csv(out / "per_client_runs.csv", index=False)
    md = ["# E3 controls", "",
          "## Reproduction checks (should be 0 or below 1e-6)", "", _md(checks), "",
          "## Federated-centralized gap under symmetric early stopping (budget 2 trajectories)", "", _md(gaps), "",
          "## Per-client inference: pooled AP, mean of client APs, positive-weighted mean", "",
          _md(pc) if len(pc) else "(no data)", "",
          f"Runs analysed: {len(runs)}; problems: {len(bad)}", ""] + [f"- {b}" for b in bad]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "numbers.json").write_text(json.dumps({k: (v if isinstance(v, (int, str)) else float(v))
                                                  for k, v in N.items()}, indent=2), encoding="utf-8")
    print(_md(checks)); print(); print(_md(gaps)); print()
    print(f"per-client analysis: {len(runs)} runs, {len(bad)} problems")
    for b in bad[:10]:
        print("  PROBLEM:", b)
    print(f"written: {out / 'report.md'}")


if __name__ == "__main__":
    main()
