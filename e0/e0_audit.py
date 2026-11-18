"""E0: data and task audit for the AMLworld extension

    python e0_audit.py --data-dir <local-path> --variant HI-Small --out C:\\path\\to\\e0_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
RENAMED = [
    "ts", "from_bank", "from_acct", "to_bank", "to_acct",
    "amount_received", "recv_currency", "amount_paid",
    "pay_currency", "pay_format", "is_laundering",
]
LABEL_COLUMN = "is_laundering"
N_CLIENTS = 10
PARTITION_SEED = 0


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_bank(s: pd.Series) -> pd.Series:
    """bank codes are kept as strings; leading zeros are stripped only to match the accounts file, which stores the same code without them"""
    out = s.astype(str).str.lstrip("0")
    return out.where(out != "", "0")


def load_transactions(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    header_norm = [c.split(".")[0] for c in header]  # pandas renames duplicate 'Account' to 'Account.1'
    if header_norm != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected schema: {header}")
    dtypes = {c: str for c in header}
    for c in ("Amount Received", "Amount Paid"):
        dtypes[c] = np.float64
    dtypes["Is Laundering"] = np.int8
    df = pd.read_csv(path, dtype=dtypes)
    df.columns = RENAMED
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df["day"] = df["ts"].dt.normalize()
    df["from_bank"] = norm_bank(df["from_bank"])
    df["to_bank"] = norm_bank(df["to_bank"])
    df["src"] = df["from_bank"] + ":" + df["from_acct"]
    df["dst"] = df["to_bank"] + ":" + df["to_acct"]
    return df


def parse_patterns(path: Path) -> pd.DataFrame:
    rows, pid, ptype = [], -1, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                pid += 1
                ptype = line.split("-", 1)[1].split(":")[0].strip()
            elif line.startswith("END LAUNDERING ATTEMPT") or not line:
                continue
            else:
                p = line.split(",")
                if len(p) == 11:
                    rows.append((pid, ptype, p[0], p[1], p[2], p[3], p[4], float(p[7])))
    pat = pd.DataFrame(rows, columns=["pattern_id", "pattern_type", "ts", "from_bank", "from_acct",
                                      "to_bank", "to_acct", "amount_paid"])
    pat["ts"] = pd.to_datetime(pat["ts"], format="%Y/%m/%d %H:%M")
    pat["from_bank"] = norm_bank(pat["from_bank"])
    pat["to_bank"] = norm_bank(pat["to_bank"])
    pat["src"] = pat["from_bank"] + ":" + pat["from_acct"]
    pat["dst"] = pat["to_bank"] + ":" + pat["to_acct"]
    return pat


def bin_pack_banks(acct_per_bank: pd.Series, n_clients: int, seed: int) -> dict:
    """greedy size-balanced assignment of whole banks to clients, by account count only"""
    rng = np.random.default_rng(seed)
    banks = acct_per_bank.index.to_numpy()
    sizes = acct_per_bank.to_numpy()
    order = np.lexsort((rng.random(len(banks)), -sizes))
    load = np.zeros(n_clients, dtype=np.int64)
    assign = {}
    for i in order:
        k = int(np.argmin(load))
        assign[banks[i]] = k
        load[k] += sizes[i]
    return assign


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.drop(columns=[LABEL_COLUMN])
    out_ = t.groupby(["src", "day"]).agg(
        n_out=("amount_paid", "size"), sum_out=("amount_paid", "sum"),
        max_out=("amount_paid", "max"), n_dst=("dst", "nunique"))
    in_ = t.groupby(["dst", "day"]).agg(
        n_in=("amount_received", "size"), sum_in=("amount_received", "sum"),
        max_in=("amount_received", "max"), n_src=("src", "nunique"))
    out_.index.names = ["acct", "day"]
    in_.index.names = ["acct", "day"]
    f = out_.join(in_, how="outer").fillna(0.0)
    return f


def leakage_test(tx: pd.DataFrame) -> dict:
    """features must be identical when the label column is permuted"""
    sample_days = sorted(tx["day"].unique())[:2]
    s = tx[tx["day"].isin(sample_days)].copy()
    f1 = build_features(s)
    s2 = s.copy()
    s2[LABEL_COLUMN] = np.random.default_rng(1).permutation(s2[LABEL_COLUMN].to_numpy())
    f2 = build_features(s2)
    same = bool(f1.equals(f2))
    return {"label_permutation_invariant": same, "rows_tested": int(len(s))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--variant", default="HI-Small")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    d = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f_tx = d / f"{args.variant}_Trans.csv"
    f_acc = d / f"{args.variant}_accounts.csv"
    f_pat = d / f"{args.variant}_Patterns.txt"
    R: dict = {"variant": args.variant}

    # 10. provenance
    R["provenance"] = {
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in (f_tx, f_acc, f_pat)},
        "code_sha256": sha256_file(Path(__file__)),
        "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
        "platform": platform.platform(),
    }
    print("hashes done", round(time.time() - t0), "s", flush=True)

    # 1. schema
    tx = load_transactions(f_tx)
    R["schema"] = {"columns_ok": True, "n_rows": int(len(tx)),
                   "label_values": sorted(tx[LABEL_COLUMN].unique().tolist()),
                   "null_cells": int(tx.isna().sum().sum())}
    print("loaded", len(tx), "transactions", round(time.time() - t0), "s", flush=True)

    acc = pd.read_csv(f_acc, dtype=str)
    acc["bank"] = norm_bank(acc["Bank ID"])
    acc["key"] = acc["bank"] + ":" + acc["Account Number"]
    tx_accts = pd.Index(pd.concat([tx["src"], tx["dst"]]).unique())
    raw_acct_banks = pd.concat([tx[["from_acct", "from_bank"]].set_axis(["a", "b"], axis=1),
                                tx[["to_acct", "to_bank"]].set_axis(["a", "b"], axis=1)]).drop_duplicates()
    multi = raw_acct_banks.groupby("a")["b"].nunique()
    R["keys"] = {
        "accounts_in_tx": int(len(tx_accts)),
        "account_numbers_in_multiple_banks": int((multi > 1).sum()),
        "accounts_file_rows": int(len(acc)),
        "accounts_file_unique_keys": int(acc["key"].nunique()),
        "tx_accounts_missing_from_accounts_file": int((~tx_accts.isin(acc["key"])).sum()),
        "entities_with_accounts_in_several_banks": int((acc.groupby("Entity ID")["bank"].nunique() > 1).sum()),
    }

    days = sorted(tx["day"].unique())
    per_day = tx.groupby("day").agg(n_tx=(LABEL_COLUMN, "size"), n_laundering=(LABEL_COLUMN, "sum"))
    R["stats"] = {
        "n_transactions": int(len(tx)), "n_banks": int(pd.concat([tx["from_bank"], tx["to_bank"]]).nunique()),
        "n_days": len(days), "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
        "self_loops": int((tx["src"] == tx["dst"]).sum()),
        "currency_conversion_tx": int((tx["recv_currency"] != tx["pay_currency"]).sum()),
        "payment_formats": tx["pay_format"].value_counts().to_dict(),
        "per_day": {str(k.date()): {"n_tx": int(v.n_tx), "n_laundering": int(v.n_laundering)}
                    for k, v in per_day.iterrows()},
    }

    # 4. labels at transaction and account-day level
    pos_src = tx.loc[tx[LABEL_COLUMN] == 1, ["src", "day"]].set_axis(["acct", "day"], axis=1)
    pos_dst = tx.loc[tx[LABEL_COLUMN] == 1, ["dst", "day"]].set_axis(["acct", "day"], axis=1)
    feats = build_features(tx)
    pos_keys = pd.MultiIndex.from_frame(pd.concat([pos_src, pos_dst]).drop_duplicates())
    y = feats.index.isin(pos_keys).astype(np.int8)
    nodes = feats.index.to_frame(index=False)
    nodes["y"] = y
    nodes["bank"] = nodes["acct"].str.split(":").str[0]
    R["labels"] = {
        "tx_positive_rate": float(tx[LABEL_COLUMN].mean()),
        "account_days": int(len(nodes)), "account_day_positive_rate": float(nodes["y"].mean()),
        "accounts_ever_positive": int(nodes.loc[nodes["y"] == 1, "acct"].nunique()),
        "positive_account_days_per_day": nodes.groupby("day")["y"].sum().rename(lambda x: str(x.date())).astype(int).to_dict(),
    }
    print("account-day table", len(nodes), round(time.time() - t0), "s", flush=True)

    # 5. intra- vs inter-bank
    inter = tx["from_bank"] != tx["to_bank"]
    R["bank_edges"] = {
        "inter_bank_share": float(inter.mean()),
        "inter_bank_share_laundering": float(inter[tx[LABEL_COLUMN] == 1].mean()),
        "inter_bank_share_licit": float(inter[tx[LABEL_COLUMN] == 0].mean()),
    }

    # 6. bank-to-client partition and cut edges
    acct_per_bank = nodes.drop_duplicates("acct").groupby("bank").size().sort_values(ascending=False)
    assign = bin_pack_banks(acct_per_bank, N_CLIENTS, PARTITION_SEED)
    cf = tx["from_bank"].map(assign)
    ct = tx["to_bank"].map(assign)
    cut = cf != ct
    nodes["client"] = nodes["bank"].map(assign)
    # account-days with no edge left inside their client on that day
    e = pd.concat([tx.loc[~cut, ["src", "day"]].set_axis(["acct", "day"], axis=1),
                   tx.loc[~cut, ["dst", "day"]].set_axis(["acct", "day"], axis=1)]).drop_duplicates()
    has_local = pd.MultiIndex.from_frame(nodes[["acct", "day"]]).isin(pd.MultiIndex.from_frame(e))
    pos_cut_src = tx.loc[(tx[LABEL_COLUMN] == 1) & cut, ["src", "day"]].set_axis(["acct", "day"], axis=1)
    pos_cut_dst = tx.loc[(tx[LABEL_COLUMN] == 1) & cut, ["dst", "day"]].set_axis(["acct", "day"], axis=1)
    pos_cut = pd.MultiIndex.from_frame(pd.concat([pos_cut_src, pos_cut_dst]).drop_duplicates())
    pos_mask = nodes["y"] == 1
    client_sizes = nodes.groupby("client").agg(account_days=("y", "size"), positives=("y", "sum"))
    R["partition_bank_binpack"] = {
        "n_clients": N_CLIENTS, "seed": PARTITION_SEED, "uses_labels": False,
        "banks_per_client": pd.Series(assign).value_counts().sort_index().astype(int).tolist(),
        "client_account_days": client_sizes["account_days"].astype(int).tolist(),
        "client_positive_rate": (client_sizes["positives"] / client_sizes["account_days"]).round(5).tolist(),
        "severed_edge_share": float(cut.mean()),
        "severed_edge_share_laundering": float(cut[tx[LABEL_COLUMN] == 1].mean()),
        "account_days_without_local_edge": float(1 - has_local.mean()),
        "positive_account_days_touching_a_severed_laundering_edge": float(
            pd.MultiIndex.from_frame(nodes.loc[pos_mask, ["acct", "day"]]).isin(pos_cut).mean()),
        "largest_bank_account_share": float(acct_per_bank.iloc[0] / acct_per_bank.sum()),
        "banks_with_1000_plus_accounts": int((acct_per_bank >= 1000).sum()),
    }

    cum = per_day["n_tx"].cumsum() / per_day["n_tx"].sum()
    train_days = cum.index[cum <= 0.6]
    val_days = cum.index[(cum > 0.6) & (cum <= 0.8)]
    test_days = cum.index[cum > 0.8]
    split = np.select([nodes["day"].isin(train_days), nodes["day"].isin(val_days)], ["train", "val"], "test")
    nodes["split"] = split
    sp = nodes.groupby("split").agg(n=("y", "size"), pos=("y", "sum"))
    R["temporal_split"] = {
        "rule": "60/20/20 of transactions by day",
        "train_days": [str(x.date()) for x in train_days],
        "val_days": [str(x.date()) for x in val_days],
        "test_days": [str(x.date()) for x in test_days],
        "sizes": {k: {"account_days": int(v.n), "positives": int(v.pos), "rate": float(v.pos / v.n)} for k, v in sp.iterrows()},
        "edges_crossing_days": 0,
    }

    # 8. account overlap between splits
    tr_accts = set(nodes.loc[nodes["split"] == "train", "acct"])
    te = nodes[nodes["split"] == "test"]
    seen = te["acct"].isin(tr_accts)
    R["account_overlap"] = {
        "test_account_days_with_account_seen_in_train": float(seen.mean()),
        "test_positives_with_account_seen_in_train": float(seen[te["y"] == 1].mean()) if (te["y"] == 1).any() else None,
        "unseen_test_account_days": int((~seen).sum()),
        "unseen_test_positives": int(te.loc[~seen, "y"].sum()),
    }

    # 9. leakage controls
    R["leakage"] = leakage_test(tx)
    R["leakage"]["feature_columns"] = feats.columns.tolist()
    R["leakage"]["patterns_used_as_features"] = False

    pat = parse_patterns(f_pat)
    pat["client_from"] = pat["from_bank"].map(assign)
    pat["client_to"] = pat["to_bank"].map(assign)
    per_pat = pat.groupby(["pattern_id", "pattern_type"]).apply(
        lambda g: pd.Series({
            "n_tx": len(g),
            "n_banks": pd.concat([g["from_bank"], g["to_bank"]]).nunique(),
            "n_clients": pd.concat([g["client_from"], g["client_to"]]).nunique(),
            "cut_share": float((g["client_from"] != g["client_to"]).mean()),
        }), include_groups=False).reset_index()
    by_type = per_pat.groupby("pattern_type").agg(
        attempts=("n_tx", "size"), mean_tx=("n_tx", "mean"), mean_banks=("n_banks", "mean"),
        mean_clients=("n_clients", "mean"), mean_cut_share=("cut_share", "mean"),
        single_client=("n_clients", lambda s: float((s == 1).mean())))
    key_pat = set(zip(pat["src"], pat["dst"], pat["ts"]))
    key_tx = set(zip(tx.loc[tx[LABEL_COLUMN] == 1, "src"], tx.loc[tx[LABEL_COLUMN] == 1, "dst"],
                     tx.loc[tx[LABEL_COLUMN] == 1, "ts"]))
    R["patterns"] = {
        "attempts": int(per_pat["pattern_id"].nunique()), "pattern_tx": int(len(pat)),
        "laundering_tx_covered_by_patterns": float(len(key_pat & key_tx) / max(1, len(key_tx))),
        "pattern_tx_found_in_tx_file": float(len(key_pat & key_tx) / max(1, len(key_pat))),
        "by_type": by_type.round(4).reset_index().to_dict(orient="records"),
    }

    R["runtime_s"] = round(time.time() - t0, 1)
    (out / "e0_report.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    nodes[["acct", "day", "y", "client", "split"]].to_parquet(out / "account_days.parquet") if _has_parquet() else \
        nodes[["acct", "day", "y", "client", "split"]].to_csv(out / "account_days.csv.gz", index=False)
    write_summary(R, out / "e0_summary.md")
    print("E0 finished in", R["runtime_s"], "s ->", out, flush=True)
    return 0


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_summary(R: dict, path: Path) -> None:
    s, l, p, ts = R["stats"], R["labels"], R["partition_bank_binpack"], R["temporal_split"]
    lines = [
        f"# E0 audit: {R['variant']}",
        f"- transactions {s['n_transactions']:,}, banks {s['n_banks']:,}, days {s['n_days']} ({s['first_day']} to {s['last_day']})",
        f"- transaction positive rate {l['tx_positive_rate']:.5f}; account-days {l['account_days']:,}, positive rate {l['account_day_positive_rate']:.5f}",
        f"- account numbers that occur in several banks: {R['keys']['account_numbers_in_multiple_banks']}",
        f"- inter-bank share: all {R['bank_edges']['inter_bank_share']:.3f}, laundering {R['bank_edges']['inter_bank_share_laundering']:.3f}",
        f"- bank bin-packing into {p['n_clients']} clients: severed edges {p['severed_edge_share']:.3f} "
        f"(laundering {p['severed_edge_share_laundering']:.3f}); account-days without a local edge {p['account_days_without_local_edge']:.3f}",
        f"- positive account-days touching a severed laundering edge: {p['positives_touching'] if 'positives_touching' in p else p['positive_account_days_touching_a_severed_laundering_edge']:.3f}",
        f"- temporal split: train {len(ts['train_days'])} days, val {len(ts['val_days'])}, test {len(ts['test_days'])}",
        f"- test account-days whose account was seen in train: {R['account_overlap']['test_account_days_with_account_seen_in_train']:.3f}",
        f"- label permutation leaves features unchanged: {R['leakage']['label_permutation_invariant']}",
        f"- patterns: {R['patterns']['attempts']} attempts, {R['patterns']['laundering_tx_covered_by_patterns']:.3f} of laundering tx covered",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
