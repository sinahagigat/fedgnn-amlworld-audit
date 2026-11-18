# -*- coding: utf-8 -*-
"""FED-PUB (Baek et al., "Personalized Subgraph Federated Learning", ICML 2023),
ported from the official code (github.com/JinheonBaek/FED-PUB, files
models/nets.py, models/layers.py, models/fedpub/client.py, models/fedpub/server.py,
main.py and scripts/disjoint.sh) into this framework, so that data, partitions,
budgets, stopping rule, checkpoint selection, seeds and evaluation are those of the
other designs.

Kept from the official code, with its default settings and no tuning:
  * model: two masked GCN layers (128 units) and a masked linear classifier;
    masks start at one and, at evaluation time, mask entries with |m| < l1 are
    set to zero; dropout 0.5 after each hidden layer;
  * optimiser: Adam, lr 0.01, weight decay 1e-6, one optimiser per client that
    persists across rounds; loss: unweighted cross-entropy on the training nodes;
  * functional embedding: mean output of the second GCN layer (eval mode) on a
    proxy graph of five 100-node random communities (p_in 0.1, p_out 0) with
    standard-normal features, generated once per run from the run seed;
  * personalised aggregation: for client i, weights exp(3 * cos(e_i, e_j)),
    row-normalised, over every client's uploaded parameters (norm_scale 3 as in
    the official run script); a client loads its personalised parameters but
    keeps its own masks.

Notes on fidelity:
  * In the official client, the L1 mask penalty and the L2 term to the received
    weights are computed from model.state_dict(), whose tensors are detached, so
    they add a constant to the loss and have no effect on the gradients. They are
    therefore omitted here; training is unchanged.
  * The official global (size-weighted) model is only used in round 1, before any
    personalised model exists; here every client starts from the same initial
    state, which is equivalent.

Taken from this framework (identical to the other designs): number of rounds and
local steps, full-batch local steps on the client subgraph, checkpoint selection on
the mean validation AP of the clients' models after local training, the patience
rule, seeds and partitions. FED-PUB has no single global model, so it is evaluated
with per-client inference only: each client scores its own nodes with its own
selected model on its own subgraph.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

logger = logging.getLogger("fedgnn.fed.fedpub")

# Official defaults (main.py set_config, parser.py, scripts/disjoint.sh)
FEDPUB_DEFAULTS = {
    "n_dims": 128,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 1e-6,
    "l1": 1e-3,
    "norm_scale": 3.0,
    "n_proxy": 5,
    "proxy_nodes": 100,
    "proxy_p_in": 0.1,
}
MASK_KEYS = ("mask",)


# ------------------------------------------------------------------ model
def _glorot_(t: torch.Tensor) -> None:
    a = math.sqrt(6.0 / (t.size(-2) + t.size(-1)))
    with torch.no_grad():
        t.uniform_(-a, a)


class MaskedGCNConv(MessagePassing):
    """GCN layer whose weight is multiplied by a learnable mask (official MaskedGCNConv)."""

    def __init__(self, in_dim: int, out_dim: int, l1: float):
        super().__init__(aggr="add")
        self.l1 = l1
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.mask = nn.Parameter(torch.ones(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        _glorot_(self.weight)

    def _mask(self) -> torch.Tensor:
        if self.training:
            return self.mask
        return self.mask.masked_fill(self.mask.abs() < self.l1, 0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, edge_weight = gcn_norm(edge_index, None, x.size(0), False, True, dtype=x.dtype)
        x = F.linear(x, self.weight * self._mask())
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return out + self.bias

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j


class MaskedLinear(nn.Module):
    """Linear classifier with a learnable mask (official MaskedLinear)."""

    def __init__(self, in_dim: int, out_dim: int, l1: float):
        super().__init__()
        self.l1 = l1
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.mask = nn.Parameter(torch.ones(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(1, out_dim))
        nn.init.xavier_uniform_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask if self.training else self.mask.masked_fill(self.mask.abs() < self.l1, 0.0)
        return F.linear(x, self.weight * m, self.bias)


class FedPubNet(nn.Module):
    """Official MaskedGCN. forward() returns (embedding, logits) like the framework's models."""

    def __init__(self, in_dim: int, n_classes: int = 2, n_dims: int = 128, dropout: float = 0.5, l1: float = 1e-3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = MaskedGCNConv(in_dim, n_dims, l1)
        self.conv2 = MaskedGCNConv(n_dims, n_dims, l1)
        self.clsif = MaskedLinear(n_dims, n_classes, l1)

    def embed(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def forward(self, x, edge_index, boundary_idx=None):
        h = self.embed(x, edge_index)
        z = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h, self.clsif(z)


def make_proxy(in_dim: int, seed: int, n_graphs: int = 5, n_nodes: int = 100, p_in: float = 0.1):
    """Official proxy graph: n_graphs disconnected random communities, N(0,1) features."""
    g = nx.random_partition_graph([n_nodes] * n_graphs, p_in=p_in, p_out=0.0, seed=seed).to_directed()
    edges = np.array(list(g.edges()), dtype=np.int64).T
    gen = torch.Generator().manual_seed(int(seed))
    x = torch.normal(0.0, 1.0, size=(n_nodes * n_graphs, in_dim), generator=gen)
    return x, torch.as_tensor(edges, dtype=torch.long)


def personalised_weights(embs: List[np.ndarray], norm_scale: float) -> np.ndarray:
    """Official server: cosine similarity of functional embeddings, exp(scale * sim), row-normalised."""
    E = np.stack(embs).astype(np.float64)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    S = (E / np.where(n == 0, 1.0, n)) @ (E / np.where(n == 0, 1.0, n)).T
    S = np.exp(norm_scale * S)
    return S / S.sum(axis=1, keepdims=True)


def aggregate(states: List[Dict[str, torch.Tensor]], w: np.ndarray) -> Dict[str, torch.Tensor]:
    return {k: sum(float(w[j]) * states[j][k] for j in range(len(states))) for k in states[0]}


class FedPubClient:
    """One client: persistent model, persistent Adam optimiser, own masks."""

    def __init__(self, cid, x, edge_index, y, train_mask, val_mask, init_state, in_dim, n_classes, hp, device,
                 class_weight=None):
        self.cid, self.device, self.hp = cid, device, hp
        self.class_weight = class_weight.to(device) if class_weight is not None else None
        self.x, self.ei, self.y = x.to(device), edge_index.to(device), y.to(device)
        self.train_mask, self.val_mask = train_mask.to(device), val_mask.to(device)
        self.model = FedPubNet(in_dim, n_classes, hp["n_dims"], hp["dropout"], hp["l1"]).to(device)
        self.model.load_state_dict(copy.deepcopy(init_state))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        self.n_nodes = int(self.x.size(0))

    def receive(self, state: Optional[Dict[str, torch.Tensor]]):
        if state is None:
            return
        own = self.model.state_dict()
        new = {k: (own[k] if any(m in k for m in MASK_KEYS) else state[k].detach().clone()) for k in own}
        self.model.load_state_dict(new)

    def train(self, steps: int) -> float:
        self.model.train()
        loss = torch.tensor(0.0)
        for _ in range(steps):
            self.opt.zero_grad()
            _, logits = self.model(self.x, self.ei)
            loss = F.cross_entropy(logits[self.train_mask], self.y[self.train_mask], weight=self.class_weight)
            loss.backward()
            self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def scores(self) -> torch.Tensor:
        self.model.eval()
        _, logits = self.model(self.x, self.ei)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def functional_embedding(self, px, pei) -> np.ndarray:
        self.model.eval()
        return self.model.embed(px.to(self.device), pei.to(self.device)).mean(dim=0).cpu().numpy()

    def state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}


def binary_val_metrics(p1: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        return 0.0, 0.5
    return float(average_precision_score(y, p1)), float(roc_auc_score(y, p1))


# ------------------------------------------------------------------ framework runner
def run_fedpub(cfg, variant, seed, full_data, in_dim, client_datasets, t_start=None) -> Dict[str, object]:
    """FED-PUB inside the framework; returns the same artifact keys as train._run_federated
    (test_metrics is None: there is no global model)."""
    from src.data.federated_dataset import build_global_test_split
    from src.evaluate import metrics_from_scores, tuned_metrics_from_scores
    from src.train import _first, _jsonable, _resolve_fed_cfg

    fed_cfg = _resolve_fed_cfg(cfg, variant)
    device = cfg.experiment.device
    hp = dict(FEDPUB_DEFAULTS)
    n_classes = 2

    init = FedPubNet(in_dim, n_classes, hp["n_dims"], hp["dropout"], hp["l1"]).to(device)
    init_state = {k: v.detach().clone() for k, v in init.state_dict().items()}
    px, pei = make_proxy(in_dim, seed, hp["n_proxy"], hp["proxy_nodes"], hp["proxy_p_in"])

    # Sensitivity variant (tag "fedpub_weighted_loss"): the class-weighted cross-entropy that every
    # other design uses (client.py: weight [1, min(n_neg / n_pos, max_pos_weight)] per client).
    # Default: the official unweighted cross-entropy.
    tags = getattr(variant, "tags", None) or []
    weighted = "fedpub_weighted_loss" in tags
    # Sensitivity variant (tag "fedpub_lr_1e-3"): learning rate 1e-3, the base learning rate of the
    # official code (main.py set_config) and the learning rate of every other design here; the
    # default 0.01 is the value the official code sets for Cora.
    if "fedpub_lr_1e-3" in tags:
        hp["lr"] = 1e-3
    # Reference variant (tag "fedpub_local"): the same model and optimiser without any communication
    # (the "Local" baseline of the FED-PUB paper): clients never receive aggregated parameters.
    local_only = "fedpub_local" in tags
    clients = []
    for fcd in client_datasets:
        d = fcd.data
        cw = None
        if weighted:
            yt = d.y[d.train_mask]
            n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
            pw = 1.0 if n_pos == 0 else min(n_neg / max(n_pos, 1), float(fed_cfg.max_pos_weight))
            cw = torch.tensor([1.0, pw], dtype=torch.float32)
        clients.append(FedPubClient(int(fcd.client_id), d.x, d.edge_index, d.y, d.train_mask, d.val_mask,
                                    init_state, in_dim, n_classes, hp, device, class_weight=cw))

    personal: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(clients)
    best_val, best_round, since = -1.0, -1, 0
    best_states = [c.state() for c in clients]
    es_states, es_best_round, es_stop_round, converged = None, -1, -1, False
    history = []
    for r in range(1, int(fed_cfg.num_rounds) + 1):
        losses, vaps, vaucs, embs, states = [], [], [], [], []
        for i, c in enumerate(clients):
            c.receive(personal[i])
            losses.append(c.train(int(fed_cfg.local_epochs)))
            p = c.scores()[:, 1].cpu().numpy()
            vm = c.val_mask.cpu().numpy()
            ap, auc = binary_val_metrics(p[vm], c.y.cpu().numpy()[vm])
            vaps.append(ap); vaucs.append(auc)
            embs.append(c.functional_embedding(px, pei))
            states.append(c.state())
        if not local_only:
            W = personalised_weights(embs, hp["norm_scale"])
            personal = [aggregate(states, W[i]) for i in range(len(clients))]

        avg_val = float(np.mean(vaps))
        history.append({"round_idx": r, "avg_train_loss": float(np.mean(losses)), "avg_val_ap": avg_val,
                        "avg_val_auc": float(np.mean(vaucs)),
                        "per_client_val_ap": {int(c.cid): float(a) for c, a in zip(clients, vaps)},
                        "selection_scores": None})
        if avg_val > best_val + fed_cfg.min_delta:
            best_val, best_round, since = avg_val, r, 0
            best_states = [copy.deepcopy(s) for s in states]
        else:
            since += 1
        if since >= fed_cfg.patience:
            converged = True
            if es_states is None:
                es_states = [copy.deepcopy(s) for s in best_states]
                es_best_round, es_stop_round = best_round, r
        logger.info(f"server | round {r}/{fed_cfg.num_rounds} | train_loss={np.mean(losses):.4f} | "
                    f"val_ap={avg_val:.4f} | best_val_ap={best_val:.4f} @ round {best_round} | "
                    f"patience={since}/{fed_cfg.patience} | converged={converged}")
        if converged and fed_cfg.early_stop_enabled:
            logger.info(f"early stop at round {r}")
            break
    train_time = time.time() - (t_start if t_start is not None else time.time())

    gtest = _first(build_global_test_split(full_data, cfg=cfg.data, seed=seed))
    y = gtest.y.detach().cpu().numpy()
    vm = gtest.val_mask.detach().cpu().numpy()
    tm = gtest.test_mask.detach().cpu().numpy()

    def pooled(stateset):
        n = sum(int(f.num_nodes) for f in client_datasets)
        out = np.full(n, np.nan)
        for c, f, s in zip(clients, client_datasets, stateset):
            c.model.load_state_dict(s)
            out[np.asarray(f.global_node_idx)] = c.scores()[:, 1].cpu().numpy()
        assert not np.isnan(out).any(), "per-client inference did not cover every node"
        return out

    p_loc = pooled(best_states)
    p_es = pooled(es_states) if es_states is not None else p_loc
    out = {
        "algorithm": "fedpub",
        "inference": "per_client_only",
        "fedpub_hyperparameters": dict(hp),
        "fedpub_loss": "class_weighted_ce" if weighted else "ce (official)",
        "fedpub_communication": "none (local training)" if local_only else "personalised aggregation (official)",
        "best_val_ap": float(best_val),
        "best_round_idx": int(best_round),
        "test_metrics": None,
        "test_metrics_es": None,
        "test_metrics_local": _jsonable(metrics_from_scores(y[tm], p_loc[tm])),
        "test_metrics_local_es": _jsonable(metrics_from_scores(y[tm], p_es[tm])),
        "es_best_round_idx": int(es_best_round if es_states is not None else best_round),
        "es_stop_round": int(es_stop_round),
        "converged": bool(converged),
        "total_rounds_run": len(history),
        "train_time_sec": round(train_time, 2),
        "num_parameters": int(sum(v.numel() for k, v in init_state.items() if "mask" not in k)),
        "num_mask_parameters": int(sum(v.numel() for k, v in init_state.items() if "mask" in k)),
        "history": history if cfg.experiment.save_per_round_history else [],
    }
    if cfg.experiment.eval_tuned_threshold:
        out["test_metrics_local_tuned"] = _jsonable(tuned_metrics_from_scores(y[vm], p_loc[vm], y[tm], p_loc[tm]))
    out["_scores"] = {"y_val": y[vm].astype(np.int8), "y_test": y[tm].astype(np.int8),
                      "p_val_local": p_loc[vm].astype(np.float32), "p_test_local": p_loc[tm].astype(np.float32),
                      "p_test_local_es": p_es[tm].astype(np.float32)}
    return out
