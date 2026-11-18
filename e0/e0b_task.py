"""E0b: task definition checks that follow from the E0 report

    python e0b_task.py --data-dir <local-path> --out C:\\...\\e0b_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_audit import (LABEL_COLUMN, bin_pack_banks, load_transactions, norm_bank,  # noqa: E402
                      parse_patterns, sha256_file)

N_CLIENTS = 10


def bin_pack(sizes: pd.Series, n_clients: int, seed: int) -> dict:
    return bin_pack_banks(sizes.sort_values(ascending=False), n_clients, seed)


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.drop(columns=[LABEL_COLUMN]).copy()
    t["hour"] = t["ts"].dt.hour.astype(np.float32)
    t["inter"] = (t["from_bank"] != t["to_bank"]).astype(np.float32)
    t["fx"] = (t["recv_currency"] != t["pay_currency"]).astype(np.float32)
    t["log_paid"] = np.log1p(t["amount_paid"]).astype(np.float32)
    t["log_recv"] = np.log1p(t["amount_received"]).astype(np.float32)
    out_ = t.groupby(["src", "day"]).agg(
        n_out=("log_paid", "size"), mean_out=("log_paid", "mean"), max_out=("log_paid", "max"),
        std_out=("log_paid", "std"), n_dst=("dst", "nunique"), inter_out=("inter", "mean"),
        fx_out=("fx", "mean"), hour_out=("hour", "mean"), ncur_out=("pay_currency", "nunique"))
    in_ = t.groupby(["dst", "day"]).agg(
        n_in=("log_recv", "size"), mean_in=("log_recv", "mean"), max_in=("log_recv", "max"),
        std_in=("log_recv", "std"), n_src=("src", "nunique"), inter_in=("inter", "mean"),
        fx_in=("fx", "mean"), hour_in=("hour", "mean"), ncur_in=("recv_currency", "nunique"))
    fmt = pd.crosstab([t["src"], t["day"]], t["pay_format"]).add_prefix("fmt_out_")
    out_.index.names = in_.index.names = fmt.index.names = ["acct", "day"]
    f = out_.join(fmt, how="left").join(in_, how="outer").fillna(0.0).astype(np.float32)
    f["log_n_out"] = np.log1p(f["n_out"])
    f["log_n_in"] = np.log1p(f["n_in"])
    return f


def leakage_test(tx: pd.DataFrame) -> bool:
    days = sorted(tx["day"].unique())[:2]
    s = tx[tx["day"].isin(days)]
    s2 = s.copy()
    s2[LABEL_COLUMN] = np.random.default_rng(1).permutation(s2[LABEL_COLUMN].to_numpy())
    return bool(build_features(s).equals(build_features(s2)))


def partition_metrics(tx, nodes, acct_client: pd.Series, pat) -> dict:
    cf = tx["src"].map(acct_client)
    ct = tx["dst"].map(acct_client)
    cut = (cf != ct).to_numpy()
    lab = (tx[LABEL_COLUMN] == 1).to_numpy()
    keep = tx.loc[~cut]
    e = pd.concat([keep[["src", "day"]].set_axis(["acct", "day"], axis=1),
                   keep[["dst", "day"]].set_axis(["acct", "day"], axis=1)]).drop_duplicates()
    nd = pd.MultiIndex.from_frame(nodes[["acct", "day"]])
    has_local = nd.isin(pd.MultiIndex.from_frame(e))
    pc = tx.loc[lab & cut]
    pos_cut = pd.MultiIndex.from_frame(pd.concat([
        pc[["src", "day"]].set_axis(["acct", "day"], axis=1),
        pc[["dst", "day"]].set_axis(["acct", "day"], axis=1)]).drop_duplicates())
    pos = nodes["y"].to_numpy() == 1
    cl = nodes["acct"].map(acct_client)
    sizes = cl.value_counts().sort_index()
    rates = nodes.groupby(cl)["y"].mean()
    p = pat.copy()
    p["cf"] = p["src"].map(acct_client)
    p["ct"] = p["dst"].map(acct_client)
    p["cut"] = p["cf"] != p["ct"]
    g = p.groupby(["pattern_id", "pattern_type"])
    per = pd.DataFrame({
        "cut_share": g["cut"].mean(),
        "n_clients": g.apply(lambda d: pd.concat([d["cf"], d["ct"]]).nunique(), include_groups=False),
    }).reset_index()
    by_type = per.groupby("pattern_type").agg(mean_cut_share=("cut_share", "mean"),
                                              mean_clients=("n_clients", "mean"),
                                              single_client=("n_clients", lambda s: float((s == 1).mean())))
    return {
        "severed_edge_share": float(cut.mean()),
        "severed_edge_share_laundering": float(cut[lab].mean()),
        "account_days_without_local_edge": float(1 - has_local.mean()),
        "positive_account_days_touching_severed_laundering_edge": float(nd[pos].isin(pos_cut).mean()),
        "client_size_min_max": [int(sizes.min()), int(sizes.max())],
        "client_positive_rate_min_max": [float(rates.min()), float(rates.max())],
        "patterns_by_type": by_type.round(4).reset_index().to_dict(orient="records"),
        "patterns_single_client_share": float((per["n_clients"] == 1).mean()),
    }


def louvain_partition(tx: pd.DataFrame, accounts: pd.Index, seed: int) -> pd.Series:
    import networkx as nx
    pairs = tx[["src", "dst"]]
    pairs = pairs[pairs["src"] != pairs["dst"]]
    s = pairs["src"].to_numpy(dtype=object)
    t = pairs["dst"].to_numpy(dtype=object)
    lo = s < t
    w = pd.DataFrame({"a": np.where(lo, s, t), "b": np.where(lo, t, s)}) \
        .groupby(["a", "b"]).size().reset_index(name="w")
    G = nx.Graph()
    G.add_nodes_from(accounts)
    G.add_weighted_edges_from(w.itertuples(index=False, name=None))
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comm_of = {}
    for i, c in enumerate(comms):
        for a in c:
            comm_of[a] = i
    comm = pd.Series(comm_of)
    assign = bin_pack(comm.value_counts(), N_CLIENTS, seed)
    return comm.map(assign)


def time_models(tx, nodes, feats, acct_client, threads: int) -> dict:
    import torch
    from torch_geometric.nn import GCNConv
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    idx = pd.Series(np.arange(len(nodes)), index=pd.MultiIndex.from_frame(nodes[["acct", "day"]]))
    s = idx.reindex(pd.MultiIndex.from_frame(tx[["src", "day"]])).to_numpy()
    d = idx.reindex(pd.MultiIndex.from_frame(tx[["dst", "day"]])).to_numpy()
    ei = np.unique(np.stack([np.concatenate([s, d]), np.concatenate([d, s])]), axis=1)
    ei = ei[:, ei[0] != ei[1]]
    x = torch.tensor(feats.to_numpy(), dtype=torch.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.tensor(nodes["y"].to_numpy(), dtype=torch.long)
    tr = torch.tensor((nodes["split"] == "train").to_numpy())

    class GCN(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.c1, self.c2 = GCNConv(f, 64), GCNConv(64, 64)
            self.n1, self.n2 = torch.nn.LayerNorm(64), torch.nn.LayerNorm(64)
            self.o = torch.nn.Linear(64, 2)

        def forward(self, x, e):
            h = torch.relu(self.n1(self.c1(x, e)))
            h = torch.relu(self.n2(self.c2(h, e)))
            return self.o(h)

    class MLP(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.m = torch.nn.Sequential(torch.nn.Linear(f, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                         torch.nn.ReLU(), torch.nn.Linear(64, 2))

        def forward(self, x, e):
            return self.m(x)

    def bench(model_cls, x, e, y, m, steps=5):
        model = model_cls(x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ts = []
        for _ in range(steps + 1):
            t0 = time.perf_counter()
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x, e)[m], y[m])
            loss.backward()
            opt.step()
            ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x, e)
        return {"train_step_s": float(np.median(ts[1:])), "eval_s": time.perf_counter() - t0}

    e_full = torch.tensor(ei, dtype=torch.long)
    R = {"threads": threads, "nodes": int(len(nodes)), "undirected_edges": int(ei.shape[1] // 2),
         "full_gcn": bench(GCN, x, e_full, y, tr), "full_mlp": bench(MLP, x, e_full, y, tr)}
    cl = nodes["acct"].map(acct_client).to_numpy()
    mask = cl == 0
    sub = np.flatnonzero(mask)
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[sub] = np.arange(len(sub))
    keep = mask[ei[0]] & mask[ei[1]]
    e_c = torch.tensor(remap[ei[:, keep]], dtype=torch.long)
    R["client0_gcn"] = bench(GCN, x[sub], e_c, y[sub], tr[sub])
    R["client0_nodes"] = int(len(sub))
    R["client0_edges"] = int(keep.sum() // 2)
    # one federated round = one local step on each of the ten clients + one evaluation pass
    R["est_fed_round_s"] = R["client0_gcn"]["train_step_s"] * N_CLIENTS + R["full_gcn"]["eval_s"]
    R["est_300_rounds_min"] = R["est_fed_round_s"] * 300 / 60
    R["est_central_1500_steps_min"] = (R["full_gcn"]["train_step_s"] + R["full_gcn"]["eval_s"]) * 1500 / 60
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--variant", default="HI-Small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--last-day", default="2022-09-10")
    ap.add_argument("--train-days", type=int, default=6)
    ap.add_argument("--val-days", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-louvain", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    d, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {"code_sha256": sha256_file(Path(__file__)), "args": vars(args)}

    tx_all = load_transactions(d / f"{args.variant}_Trans.csv")
    per_day = tx_all.groupby("day").agg(n=(LABEL_COLUMN, "size"), pos=(LABEL_COLUMN, "sum"))
    last = pd.Timestamp(args.last_day)
    tail = per_day[per_day.index > last]
    R["tail_removed"] = {
        "days": [str(x.date()) for x in tail.index], "transactions": int(tail["n"].sum()),
        "laundering_tx": int(tail["pos"].sum()),
        "laundering_share_in_tail": float(tail["pos"].sum() / max(1, tail["n"].sum())),
        "laundering_share_kept": float(per_day.loc[per_day.index <= last, "pos"].sum()
                                       / per_day.loc[per_day.index <= last, "n"].sum()),
    }
    tx = tx_all[tx_all["day"] <= last].reset_index(drop=True)
    del tx_all
    print("tail removed", R["tail_removed"]["transactions"], "tx;", round(time.time() - t0), "s", flush=True)

    feats = build_features(tx)
    lab = tx[tx[LABEL_COLUMN] == 1]
    pos = pd.MultiIndex.from_frame(pd.concat([lab[["src", "day"]].set_axis(["acct", "day"], axis=1),
                                              lab[["dst", "day"]].set_axis(["acct", "day"], axis=1)]).drop_duplicates())
    nodes = feats.index.to_frame(index=False)
    nodes["y"] = feats.index.isin(pos).astype(np.int8)
    nodes["bank"] = nodes["acct"].str.split(":").str[0]
    days = sorted(nodes["day"].unique())
    tr_d, va_d = days[:args.train_days], days[args.train_days:args.train_days + args.val_days]
    nodes["split"] = np.select([nodes["day"].isin(tr_d), nodes["day"].isin(va_d)], ["train", "val"], "test")
    sp = nodes.groupby("split").agg(n=("y", "size"), pos=("y", "sum"))
    R["split"] = {"train_days": [str(x.date()) for x in tr_d], "val_days": [str(x.date()) for x in va_d],
                  "test_days": [str(x.date()) for x in days[args.train_days + args.val_days:]],
                  "sizes": {k: {"account_days": int(v.n), "positives": int(v.pos), "rate": float(v.pos / v.n)}
                            for k, v in sp.iterrows()}}
    tr_accts = set(nodes.loc[nodes["split"] == "train", "acct"])
    te = nodes[nodes["split"] == "test"]
    seen = te["acct"].isin(tr_accts)
    R["split"]["test_unseen_account_days"] = int((~seen).sum())
    R["split"]["test_unseen_positives"] = int(te.loc[~seen, "y"].sum())
    R["features"] = {"columns": feats.columns.tolist(), "n": int(feats.shape[1]),
                     "label_permutation_invariant": leakage_test(tx)}
    print("features", feats.shape, round(time.time() - t0), "s", flush=True)

    acc = pd.read_csv(d / f"{args.variant}_accounts.csv", dtype=str)
    acc["bank"] = norm_bank(acc["Bank ID"])
    country = acc["Bank Name"].str.extract(r"^(.*) Bank #\d+$")[0]
    acc["group"] = country.fillna("bank:" + acc["bank"])
    bank_group = acc.drop_duplicates("bank").set_index("bank")["group"]
    accounts = pd.Index(nodes["acct"].unique())
    acct_bank = pd.Series(accounts.str.split(":").str[0], index=accounts)
    acct_per_bank = acct_bank.value_counts()
    pat = parse_patterns(d / f"{args.variant}_Patterns.txt")
    pat = pat[pat["ts"].dt.normalize() <= last]

    P = {}
    for seed in range(3):
        a = bin_pack(acct_per_bank, N_CLIENTS, seed)
        P[f"bank_binpack_s{seed}"] = acct_bank.map(a)
    grp_sizes = acct_bank.map(bank_group).fillna("unknown").value_counts()
    a = bin_pack(grp_sizes, N_CLIENTS, 0)
    P["country_binpack_s0"] = acct_bank.map(bank_group).fillna("unknown").map(a)
    rng = np.random.default_rng(0)
    P["random_accounts_s0"] = pd.Series(rng.integers(0, N_CLIENTS, len(accounts)), index=accounts)
    if not args.skip_louvain:
        t1 = time.time()
        P["louvain_s0"] = louvain_partition(tx, accounts, 0)
        R["louvain_runtime_s"] = round(time.time() - t1, 1)
    R["country_groups"] = {"n_groups": int(len(grp_sizes)), "largest_share": float(grp_sizes.iloc[0] / grp_sizes.sum()),
                           "named_country_share": float(acct_bank.map(bank_group).fillna("bank:").str.startswith("bank:").eq(False).mean())}
    R["partitions"] = {}
    for name, ac in P.items():
        R["partitions"][name] = partition_metrics(tx, nodes, ac, pat)
        print("partition", name, round(R["partitions"][name]["severed_edge_share"], 3), round(time.time() - t0), "s", flush=True)

    if not args.skip_timing:
        R["timing"] = time_models(tx, nodes, feats, P["bank_binpack_s0"], args.threads)
        print("timing", json.dumps(R["timing"]), flush=True)

    R["runtime_s"] = round(time.time() - t0, 1)
    (out / "e0b_report.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    print("E0b finished in", R["runtime_s"], "s ->", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
