# E1: FED-PUB on Elliptic

This folder is a copy of the paper's code (taken from `../e2`, whose Elliptic path is the code used
in the paper) with one published method added: FED-PUB (Baek et al., ICML 2023), ported from the
authors' repository into `src/fed/fedpub.py`. We kept its default hyperparameters and did not tune
it. Apart from the new file, the only changes are one dispatch line in
`src/train.py::_run_federated`, `"fedpub"` in `src/config.py::SUPPORTED_ALGORITHMS`, the data path in
`tools/run_local.py` and the job list in `tools/jobs.py`. All other designs run exactly as before.

FED-PUB keeps one model per client, so it can only be evaluated per client. In its run records
`test_metrics` is therefore null, and the results are in `test_metrics_local` and
`test_metrics_local_es`.

## Protocols

Each protocol uses the 20 seeds 42-61 and the paper's configs, with FedPub_GCN as the only variant.

- PA: Louvain partitions redrawn per seed, budget 1 (100 rounds of one step) with early stopping, as protocol A in the paper.
- PN: redrawn Louvain partitions, 300 rounds of five steps without early stopping, as protocol N.
- PQ0-PQ9: the paper's fixed partitions C0-C9 (partition seeds 0-4 and 42-46) under the full budget, as protocol Q.

Three sensitivity runs change one thing in PN:

- PW uses the class-weighted loss of the other designs.
- PL uses learning rate 1e-3, the base value of the official code and of our designs. The default of 0.01 is what the official code sets for Cora.
- PO trains each client alone, without communication, with the FED-PUB model and optimiser.

## Checking the port

`python tools/fedpub_cora_check.py --runs 5` reproduces the official non-overlapping Cora setting
with ten clients from a METIS partition. If pymetis will not install, the script falls back to the
METIS membership saved in `tools/cora_metis_10.json`, computed once with pymetis 2025.2.2 on the same
graph and checked against an edge-list hash. On our workstation, over five runs, test accuracy at the
best validation round was 82.54 +- 0.39 for FED-PUB and 81.31 +- 0.21 for local training, against
81.54 +- 0.12 and 79.94 +- 0.24 in the paper by Baek et al. The output is in
`analysis/output/cora_check.txt`.

## Stopping-rule controls

These runs belong to Section 4.7 and Table S35 (stopping rule) and Section 4.11 and Table S36
(per-client mean AP).

- CE1-CE4 train the centralized GCN and MLP for 1,500 steps (budget 2) and record, next to the final checkpoint, the one a patience of 20 steps would have returned.
- CH1-CH4 do the same with a patience of 100 steps, which matches 20 rounds of five local steps.
- FN1-FN4 rerun FedAvg (GCN and MLP) under protocol N, saving per-node predictions and the patience-20 checkpoint of the same run.

`analysis/e3_analysis.py` computes the federated-centralized gap under symmetric early stopping and
the per-client mean AP. For the latter it rebuilds each partition and checks it against the run
record first.
