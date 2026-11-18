"""build the AMLworld account-day graph once and save it in the format the training code expects (same fields as the Elliptic graph, plus bank_id and country_id for the institutional partitions)

    python tools/build_aml_graph.py --aml-dir <local-path> --out-dir <local-path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

LAST_DAY = "2022-09-10"
TRAIN_DAYS, VAL_DAYS = 6, 2
LABEL = "is_laundering"
COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_received", "recv_currency",
        "amount_paid", "pay_currency", "pay_format", LABEL]


def norm_bank(s):
    out = s.astype(str).str.lstrip("0")
    return out.where(out != "", "0")


def load(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {c: str for c in header}
    dtypes["Amount Received"] = dtypes["Amount Paid"] = np.float64
    dtypes["Is Laundering"] = np.int8
    df = pd.read_csv(path, dtype=dtypes)
    df.columns = COLS
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df["day"] = df["ts"].dt.normalize()
    df = df[df["day"] <= pd.Timestamp(LAST_DAY)].reset_index(drop=True)
    df["from_bank"] = norm_bank(df["from_bank"])
    df["to_bank"] = norm_bank(df["to_bank"])
    df["src"] = df["from_bank"] + ":" + df["from_acct"]
    df["dst"] = df["to_bank"] + ":" + df["to_acct"]
    return df


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.drop(columns=[LABEL]).copy()
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
    return f.sort_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aml-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="HI-Small")
    args = ap.parse_args()
    src, out = Path(args.aml_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tx = load(src / f"{args.variant}_Trans.csv")
    feats = build_features(tx)
    nodes = feats.index.to_frame(index=False)
    lab = tx[tx[LABEL] == 1]
    pos = pd.MultiIndex.from_frame(pd.concat([lab[["src", "day"]].set_axis(["acct", "day"], axis=1),
                                              lab[["dst", "day"]].set_axis(["acct", "day"], axis=1)]).drop_duplicates())
    y = feats.index.isin(pos).astype(np.int64)
    days = sorted(nodes["day"].unique())
    day_idx = nodes["day"].map({d: i + 1 for i, d in enumerate(days)}).to_numpy()
    train = day_idx <= TRAIN_DAYS
    val = (day_idx > TRAIN_DAYS) & (day_idx <= TRAIN_DAYS + VAL_DAYS)
    test = day_idx > TRAIN_DAYS + VAL_DAYS

    idx = pd.Series(np.arange(len(nodes)), index=feats.index)
    s = idx.reindex(pd.MultiIndex.from_frame(tx[["src", "day"]])).to_numpy()
    d = idx.reindex(pd.MultiIndex.from_frame(tx[["dst", "day"]])).to_numpy()
    ei = np.stack([np.concatenate([s, d]), np.concatenate([d, s])])
    ei = ei[:, ei[0] != ei[1]]
    ei = np.unique(ei, axis=1)

    x = feats.to_numpy(dtype=np.float32)
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    x = (x - mu) / sd

    bank = nodes["acct"].str.split(":").str[0]
    bank_codes, bank_id = np.unique(bank.to_numpy(), return_inverse=True)
    acc = pd.read_csv(src / f"{args.variant}_accounts.csv", dtype=str)
    acc["bank"] = norm_bank(acc["Bank ID"])
    grp = acc["Bank Name"].str.extract(r"^(.*) Bank #\d+$")[0].fillna("bank:" + acc["bank"])
    bank_group = pd.Series(grp.to_numpy(), index=acc["bank"]).groupby(level=0).first()
    country = pd.Series(bank_codes).map(bank_group).fillna("unknown").to_numpy()
    _, country_of_bank = np.unique(country, return_inverse=True)

    data = Data(x=torch.tensor(x), y=torch.tensor(y), edge_index=torch.tensor(ei, dtype=torch.long))
    data.time_step = torch.tensor(day_idx, dtype=torch.long)
    data.train_mask = torch.tensor(train)
    data.val_mask = torch.tensor(val)
    data.test_mask = torch.tensor(test)
    data.tx_id = np.arange(len(nodes))
    data.bank_id = bank_id.astype(np.int64)
    data.country_id = country_of_bank[bank_id].astype(np.int64)
    data.feature_names = feats.columns.tolist()
    data.meta = {
        "dataset": f"AMLworld {args.variant} account-day", "num_nodes": int(len(nodes)),
        "num_edges_final": int(ei.shape[1]), "positive_rate": float(y.mean()),
        "positive_rate_train": float(y[train].mean()), "positive_rate_val": float(y[val].mean()),
        "positive_rate_test": float(y[test].mean()), "n_train": int(train.sum()), "n_val": int(val.sum()),
        "n_test": int(test.sum()), "num_features": int(x.shape[1]), "split_mode": "temporal",
        "n_banks": int(len(bank_codes)), "n_country_groups": int(country_of_bank.max() + 1),
    }
    name = f"aml_{args.variant.lower().replace('-', '_')}_graph.pt"
    torch.save(data, out / name)
    (out / "AMLWORLD").write_text(name, encoding="utf-8")
    h = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / "graph_meta.json").write_text(json.dumps({**data.meta, "sha256": h,
                                                     "features": data.feature_names}, indent=2), encoding="utf-8")
    print(json.dumps(data.meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
