"""builds Online Resource 2 (Tables S24-S34: AMLworld and FED-PUB) as LaTeX

    python analysis/make_esm2.py --e2-out analysis/output --e1-out ../e1/analysis/output
        --out analysis/output/esm2.tex
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NAMES = {"FedAvg_GCN": "FedAvg", "FedProto_GCN": "FedProto", "FedGNN_Hard_GCN": "Hard",
         "FedGNN_Soft_GCN": "Soft", "Centralized_GCN": "Centralized GCN", "Centralized_MLP": "Centralized MLP",
         "FedAvg_MLP": "FedAvg-MLP", "FedGNN_Soft_Virtual_GCN": "Soft-VN",
         "FedGNN_Soft_Virtual_GAT": "Soft-VN-GAT", "Central-GCN": "Centralized GCN",
         "Central-MLP": "Centralized MLP", "Soft-VN-GAT": "Soft-VN-GAT"}


def esc(x):
    if pd.isna(x):
        return "--"
    x = str(x)
    for a, b in (("%", r"\%"), ("_", r"\_"), ("<", r"$<$"), ("&", r"\&")):
        x = x.replace(a, b)
    return x.replace("-0.", "$-$0.").replace("+0.", "$+$0.")


def tabular(df, size=r"\scriptsize"):
    cols = list(df.columns)
    out = [size, r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{'l' + 'r' * (len(cols) - 1)}}}",
           r"\toprule", " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join(esc(NAMES.get(r[c], r[c]) if c == cols[0] else r[c]) for c in cols) + r" \\")
    return "\n".join([r"\begin{adjustbox}{max width=\textwidth}"] + out + [r"\bottomrule", r"\end{tabular}",
                                                                          r"\end{adjustbox}"])


def tab(caption, *dfs):
    """one numbered table; several data frames are stacked inside it"""
    body = "\n\n\\medskip\n".join(tabular(d) for d in dfs)
    return "\n".join([r"\begin{table}[!htbp]", rf"\caption{{{caption}}}", r"\centering", body, r"\end{table}", ""])


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); out = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); out[k] = min(run, 1.0)
    return out


def pairwise(runs):
    designs = ["FedAvg_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN"]
    rows = []
    for exp, part in (("AB", "bank"), ("AL", "Louvain")):
        for col, inf in (("ap", "global"), ("ap_local", "per-client")):
            w = runs[runs.exp == exp].pivot(index="seed", columns="variant", values=col)
            pairs = list(itertools.combinations(designs, 2))
            ps = [stats.wilcoxon(w[a], w[b], method="exact").pvalue for a, b in pairs]
            for (a, b), p, h in zip(pairs, ps, holm(ps)):
                d = (w[a] - w[b]).values
                rows.append({"Partition": part, "Inference": inf, "Pair": f"{NAMES[a]} -- {NAMES[b]}",
                             "Difference": f"{d.mean():+.4f}", "Wins": f"{int((d > 0).sum())}/20",
                             "p": f"{p:.4f}" if p >= 1e-4 else "<0.0001",
                             "Holm p": f"{h:.4f}" if h >= 1e-4 else "<0.0001"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2-out", default="analysis/output")
    ap.add_argument("--e1-out", default="../e1/analysis/output")
    ap.add_argument("--e3-out", default="../e1/analysis/output_e3")
    ap.add_argument("--out", default="analysis/output/esm2.tex")
    a = ap.parse_args()
    e2, e1 = Path(a.e2_out) / "tables", Path(a.e1_out) / "tables"
    r = lambda p: pd.read_csv(p, dtype=str)

    T = []
    rc = r(e2 / "run_counts.csv")
    T.append(tab("AMLworld: completed runs per design and protocol (AB, AL, AC, AR: bank, Louvain, "
                 "country and random partitions at budget~1; F0--F4: fixed Louvain partitions at budget~1; "
                 "NB, NC: bank partition and centralized training at the full budget).", rc))

    parts = []
    for part, g, l in (("bank", "b1_bank_designs.csv", "local_b1_bank_designs.csv"),
                       ("Louvain", "b1_louvain_designs.csv", "local_b1_Louvain_designs.csv")):
        G, L = r(e2 / g), r(e2 / l).set_index("design")
        for _, x in G.iterrows():
            y = L.loc[x["design"]] if x["design"] in L.index else None
            parts.append({"Partition": part, "Design": NAMES.get(x["design"], x["design"]), "AP G": x["AP mean"],
                          "SD": x["AP SD"], "minus FedAvg G [95% CI]": x["minus FedAvg [95% CI]"],
                          "Holm p G": x["Holm p"],
                          "AP P": y["AP"] if y is not None else x["local AP"],
                          "minus FedAvg P [95% CI]": y["minus FedAvg [95% CI]"] if y is not None else None,
                          "Holm p P": y["Holm p"] if y is not None else None})
    T.append(tab("AMLworld, budget~1: test AP per design under global (G) and per-client (P) inference, "
                 "mean over 20 paired seeds; differences to FedAvg with Wilcoxon $p$ Holm-corrected over the "
                 "designs within each partition and inference protocol. Centralized and graph-free models use no "
                 "client subgraphs.", pd.DataFrame(parts)))

    T.append(tab("AMLworld, budget~1: pairwise comparisons among the four convolutional designs without "
                 "virtual nodes (row design minus column design, test AP), Wilcoxon signed-rank test with Holm "
                 "correction over the six pairs within each partition and inference protocol.",
                 pairwise(pd.read_csv(Path(a.e2_out) / "runs.csv"))))

    bp, lp = r(e2 / "b1_partitions.csv"), r(e2 / "local_b1_partitions.csv").set_index("partition")
    rows = [{"Partition": x["partition"], "Edges kept": x["edges kept"], "Soft $-$ FedAvg G": x["Soft minus FedAvg [95% CI]"],
             "Holm p G": x["Holm p (4 partitions)"], "Soft $-$ FedAvg P": lp.loc[x["partition"], "Soft minus FedAvg [95% CI]"],
             "Holm p P": lp.loc[x["partition"], "Holm p (4 partitions)"],
             "Centr. $-$ FedAvg G": x["Central-GCN minus FedAvg [95% CI]"],
             "Centr. $-$ FedAvg P": lp.loc[x["partition"], "Central-GCN minus FedAvg [95% CI]"]} for _, x in bp.iterrows()]
    T.append(tab("AMLworld, budget~1, partitions redrawn per seed: Soft minus FedAvg and centralized GCN "
                 "minus FedAvg (test AP, 95\\% CI) under global (G) and per-client (P) inference; Holm over the four "
                 "partition schemes.", pd.DataFrame(rows).rename(columns=lambda c: c)))

    T.append(tab("AMLworld, budget~1: Soft minus FedAvg on five fixed Louvain partitions (F0--F4), each "
                 "reused for all 20 seeds; Holm over the five partitions within each inference protocol.",
                 r(e2 / "fixed_partitions.csv")))

    fd = r(e2 / "full_designs.csv")
    fd = r(e2 / "full_designs.csv")
    fv, lf = r(e2 / "full_vs_fedavg.csv"), r(e2 / "local_full_vs_fedavg.csv")
    lf = lf[lf.design != "FedAvg"].assign(protocol="full budget, per-client")
    comb = pd.concat([fv.assign(protocol=fv.protocol + ", global"), lf.drop(columns=["AP"])])[
        ["protocol", "design", "n", "minus FedAvg [95% CI]", "wins", "Holm p"]]
    gv = pd.concat([r(e2 / "graph_value.csv"), r(e2 / "local_graph_value.csv")])
    T.append(tab("AMLworld, bank partition, full budget (300 rounds $\\times$ 5 local steps). Top: test AP under "
                 "global inference, the checkpoint the patience-20 rule would have returned (ES), the number of runs "
                 "in which the rule fires, the median round of the best validation checkpoint and the run time. "
                 "Middle: designs against FedAvg, Holm over the designs within each protocol. Bottom: value of the "
                 "graph, GCN minus MLP with the same training.", fd, comb, gv))

    T.append(tab("FED-PUB on Elliptic, fixed partitions C0--C9 (partition seeds 0--4 and 42--46), full "
                 "budget: per-client test AP over 20 paired seeds; Holm over the ten partitions.",
                 r(e1 / "fixed_partitions.csv").assign(partition=lambda d: d["partition"].str.replace("Q", "C"))))
    T.append(tab("Elliptic, protocol N (partitions redrawn per seed, full budget): per-client test AP of every "
                 "federated design against FedAvg, Holm over the designs.", r(e1 / "n_all_designs.csv")))
    T.append(tab("FED-PUB and its sensitivity variants: full-budget checkpoint against the checkpoint the "
                 "patience-20 rule would have returned, per-client test AP.", r(e1 / "early_stopping.csv")))
    T.append(tab("FED-PUB with the class-weighted loss of the other designs (protocol N): per-client test AP.",
                 r(e1 / "sensitivity_weighted_loss.csv")))
    T.append(tab("FED-PUB with learning rate $10^{-3}$ (the base learning rate of its official code and the "
                 "learning rate of the other designs; 0.01 is the value its code sets for Cora) and local training "
                 "with the FED-PUB model and optimizer but no communication between clients (protocol N): "
                 "per-client test AP over 20 paired seeds, uncorrected Wilcoxon $p$.",
                 r(e1 / "sensitivity_lr_local.csv")))
    e3 = Path(a.e3_out) / "tables"
    T.append(tab("Elliptic, budget 2 trajectories rerun with the checkpoint the patience rule would return (jobs CE, CH, "
                 "FN): federated--centralized gap (centralized minus FedAvg, test AP under global inference) when only "
                 "the federated runs stop, when both stop with a patience of 20 rounds or steps, when both stop after "
                 "the same number of local steps without improvement (20 rounds of five steps, 100 centralized "
                 "steps), and when neither stops. Bottom: reproduction of the original runs by the reruns.",
                 r(e3 / "gap_symmetric_stopping.csv").replace({"rule": {"paper: federated stops (patience 20 rounds), centralized does not": "only federated training stops (patience 20 rounds), reruns"}}), r(e3 / "reproduction_checks.csv")))
    pc = r(e3 / "per_client_mean_ap.csv")
    fp = {"FedPub_GCN": "FED-PUB", "FedPub_GCN_wloss": "FED-PUB (weighted loss)", "FedPub_GCN_lr1e3": "FED-PUB (lr 1e-3)",
          "FedPub_GCN_local": "Local", "FedAvg_GCN": "FedAvg", "FedGNN_Soft_GCN": "Soft"}
    mt = {"pooled": "pooled", "client_mean": "mean of client APs", "client_wmean": "positive-weighted mean"}
    pc = pc.assign(design=pc.design.map(lambda x: fp.get(x, x)), vs=pc.vs.map(lambda x: fp.get(x, x)),
                   metric=pc.metric.map(lambda x: mt.get(x, x)))
    T.append(tab("Elliptic, per-client inference with partitions redrawn per seed (A: budget 1; N: full budget): test "
                 "AP with the scores of all clients pooled (as in the main text), as the mean of the client APs and as "
                 "their mean weighted by the number of positive test nodes; partitions rebuilt and checked against the "
                 "run records. Holm over the comparisons within each protocol and metric.", pc))

    body = "\n".join(T)
    doc = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm,a4paper]{geometry}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{table}{23}
\begin{document}
\noindent\textbf{Online Resource 2}\\[2pt]
When the protocol picks the winner: a controlled variance audit of federated graph neural networks for illicit
transaction detection\\[2pt]
Knowledge and Information Systems\\[2pt]
Sina Haghighat, Mohammad Ali Balafar\\[2pt]
Faculty of Electrical and Computer Engineering, University of Tabriz, 29 Bahman Blvd, Tabriz 5166616471, Iran\\[2pt]
Corresponding author: Mohammad Ali Balafar, balafarila@tabrizu.ac.ir\\[6pt]
Tables S24--S36: AMLworld (Section 4.10), FED-PUB (Section 4.11) and the stopping-rule control (Section 4.7). All values are computed from the released run
records by the released analysis scripts.
""" + body + "\n\\end{document}\n"
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
