"""sanity check of the FED-PUB port (src/fed/fedpub.py) against the published result

    python tools/fedpub_cora_check.py [--data-dir <local-path>] [--runs 3]
"""
import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fed.fedpub import FEDPUB_DEFAULTS, FedPubClient, FedPubNet, aggregate, make_proxy, personalised_weights  # noqa: E402


def load_cora(data_dir):
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    data = Planetoid(str(data_dir), "Cora", transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
    torch.manual_seed(1234)
    n = data.num_nodes
    n_tr, n_te = round(n * 0.2), round(n * 0.4)
    perm = torch.randperm(n)
    for name, idx in (("train_mask", perm[:n_tr]), ("test_mask", perm[n_tr:n_tr + n_te]), ("val_mask", perm[n_tr + n_te:])):
        m = torch.zeros(n, dtype=torch.bool); m[idx] = True; setattr(data, name, m)
    return data


def _precomputed_metis(data):
    f = Path(__file__).resolve().parent / "cora_metis_10.json"
    if not f.is_file():
        return None
    import hashlib
    import json
    rec = json.loads(f.read_text(encoding="utf-8"))
    e = data.edge_index.cpu().numpy().astype(np.int64)
    e = e[:, np.lexsort((e[1], e[0]))]
    if data.num_nodes != rec["num_nodes"] or hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() != rec["edge_index_sha256"]:
        print("precomputed METIS partition ignored: the loaded graph differs from the one it was computed on")
        return None
    return np.asarray(rec["membership"]), f"METIS, precomputed ({rec['tool']}; edge cut {rec['edge_cut']})"


def partition(data, k=10):
    G = nx.Graph(); G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())
    try:
        import pymetis
        adj = [list(G.neighbors(i)) for i in range(data.num_nodes)]
        _, memb = pymetis.part_graph(k, adjacency=adj)
        return np.asarray(memb), "METIS (pymetis)"
    except ImportError:
        pre = _precomputed_metis(data)
        if pre is not None:
            return pre
        comms = sorted(nx.community.louvain_communities(G, seed=1234), key=len, reverse=True)
        load, memb = np.zeros(k, int), np.zeros(data.num_nodes, int)
        for c in comms:
            j = int(np.argmin(load)); memb[list(c)] = j; load[j] += len(c)
        return memb, "Louvain packed into 10 clients (pymetis not installed)"


def subgraph(data, nodes):
    from torch_geometric.utils import subgraph as sg
    nodes = torch.as_tensor(nodes, dtype=torch.long)
    ei, _ = sg(nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    return data.x[nodes], ei, data.y[nodes], data.train_mask[nodes], data.val_mask[nodes], data.test_mask[nodes]


def run(data, memb, seed, personalised=True, rounds=100, epochs=1):
    torch.manual_seed(seed); np.random.seed(seed)
    hp = dict(FEDPUB_DEFAULTS); nc = int(data.y.max()) + 1; d = data.num_features
    init = FedPubNet(d, nc, hp["n_dims"], hp["dropout"], hp["l1"]).state_dict()
    px, pei = make_proxy(d, seed)
    parts = [subgraph(data, np.flatnonzero(memb == c)) for c in range(int(memb.max()) + 1)]
    cl = [FedPubClient(i, x, ei, y, tr, va, init, d, nc, hp, "cpu") for i, (x, ei, y, tr, va, te) in enumerate(parts)]
    tests = [p[5] for p in parts]
    pers, best, best_acc, final = [None] * len(cl), -1.0, None, None
    for _ in range(rounds):
        val, test, embs, states = [], [], [], []
        for i, c in enumerate(cl):
            c.receive(pers[i]); c.train(epochs)
            pred = c.scores().argmax(1)
            val.append(float((pred[c.val_mask] == c.y[c.val_mask]).float().mean()))
            test.append(float((pred[tests[i]] == c.y[tests[i]]).float().mean()))
            embs.append(c.functional_embedding(px, pei)); states.append(c.state())
        if personalised:
            W = personalised_weights(embs, hp["norm_scale"])
            pers = [aggregate(states, W[i]) for i in range(len(cl))]
        if np.mean(val) > best:
            best, best_acc = float(np.mean(val)), float(np.mean(test))
        final = float(np.mean(test))
    return 100 * best_acc, 100 * final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "data" / "cora"))
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    data = load_cora(a.data_dir)
    memb, how = partition(data)
    print(f"Cora: {data.num_nodes} nodes, {data.num_edges} arcs; partition: {how}")
    for name, pers in (("FED-PUB", True), ("Local (no communication)", False)):
        r = np.array([run(data, memb, s, pers) for s in range(a.runs)])
        print(f"{name:24s} test accuracy at best validation round {r[:, 0].mean():.2f} +- {r[:, 0].std():.2f} | "
              f"final round {r[:, 1].mean():.2f} +- {r[:, 1].std():.2f}")
    print("published (10 clients, METIS): FED-PUB 81.54 +- 0.12, Local 79.94 +- 0.24")


if __name__ == "__main__":
    main()
