# fedgnn-amlworld-audit

This repository holds the code and run records behind two parts of our article

> S. Haghighat, M. A. Balafar. When the protocol picks the winner: a controlled variance audit of
> federated graph neural networks for illicit transaction detection. (submitted)

namely the experiments on a second dataset, IBM AMLworld (HI-Small), and the check of a published
method, FED-PUB, on Elliptic. Everything else in the paper, including the 7,800 Elliptic runs, lives in
[fedgnn-elliptic-audit](https://github.com/sinahagigat/fedgnn-elliptic-audit)
(DOI 10.5281/zenodo.23825265). Nothing here changes that repository.

## What is where

`e0/` is where we looked at AMLworld before running anything: dropping the tail of the transaction
period, the day-based split, partitions, account-day features and a label-permutation leakage test.
It only produces reports. The first audit (E0) used a split and partition settings that we later
replaced, so the task used in the paper is the one in `results_b/e0b_report.json` (E0b).

`e2/` contains the AMLworld experiments. It is the paper's code with bank, country, Louvain and random
partitions added, a loader for the prebuilt graph (`tools/build_aml_graph.py`), the configs for
protocols AB, AL, AC, AR, F0-F4, NB and NC, the run records (`results/<job>/artifacts.json`) and the
analysis (`analysis/e2_analysis.py`, `analysis/aml_references.py`, output in `analysis/output`).

`e1/` has the FED-PUB port (Baek et al., ICML 2023) in `src/fed/fedpub.py`, a check of the port on
Cora, and the FED-PUB runs (PA, PN, PQ0-PQ9, PW, PL, PO). The other designs these runs are compared
with are the paper's own runs in fedgnn-elliptic-audit. The stopping-rule controls CE, CH and FN, and
their analysis in `analysis/e3_analysis.py` (Tables S35 and S36), are here too. `e1/README.md` says
more.

## Data

We do not redistribute AMLworld. Get HI-Small (`HI-Small_Trans.csv`, `HI-Small_accounts.csv`,
`HI-Small_Patterns.txt`) from the AMLworld release of Altman et al. (NeurIPS 2023 Datasets and
Benchmarks, licence CDLA-Sharing-1.0) and build the graph with `e2/tools/build_aml_graph.py`.
For Elliptic, see fedgnn-elliptic-audit.

## Rerunning

The environment is in `e2/requirements.txt`. We ran everything on CPU with Python 3.14 and
deterministic algorithms; networkx has to be 3.6.1, since other versions give different Louvain
partitions. A job is started with `tools/run_local.py --jobs <JOB>` from inside `e2/` or `e1/`.
The analyses only read the run records and take a few seconds:

```
cd e2 && python analysis/e2_analysis.py
cd e1 && python analysis/e1_analysis.py --e1-results results --paper-results <elliptic results> --frozen-repo <fedgnn-elliptic-audit>
```

Per-node predictions and logs are left out to keep the repository small. `MANIFEST.sha256` has a
hash for every file.

## A few notes on the code

We left the `src/` folders exactly as they were when the runs were made, because every run record
stores a hash of that code. This means some leftovers from earlier versions are still there.
`src/fed/privacy.py` and `src/rust_ext/` are not used by any protocol. `src/compat.py` translates
config names from older versions of the code to the current ones; the configs in `configs/` already
use the current names, so it does nothing when you rerun the released jobs.

Some docstrings in `src/` are also older than the final analysis, for example the one in
`build_global_test_split` still treats global inference as the main evaluation, which the paper
no longer does. `src/stats.py` is not used for any number in the paper; all reported statistics
come from the scripts in `analysis/`.

Results repeat to the last digit only with the same number of torch threads, which is written in each
run log. With another thread count the floating-point sums come out in a different order, and for
the graph-free models that can be enough to pick a different checkpoint.

## Licence

The code is MIT licensed (see `LICENSE`). Run records and analysis outputs are CC BY 4.0.
