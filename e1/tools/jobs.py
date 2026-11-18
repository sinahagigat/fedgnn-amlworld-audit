"""E1 (FED-PUB on Elliptic) job table

    python tools/jobs.py   # print the table
"""

SEED_BLOCKS = ["42-46", "47-51", "52-56", "57-61"]
EXPERIMENTS = [
    ("PA", "configs/e1_PA_louvain_budget1.yaml"),   # protocol A: redrawn Louvain partitions budget 1, early stopping
    ("PN", "configs/e1_PN_louvain_full.yaml"),      # protocol N: redrawn Louvain partitions, 300 rounds x 5 steps, no early stopping
    ("PW", "configs/e1_PW_louvain_full_wloss.yaml"),  # sensitivity: protocol N with the class-weighted loss of the other designs
    ("PL", "configs/e1_PL_louvain_full_lr1e-3.yaml"),  # sensitivity: protocol N with learning rate 1e-3
    ("PO", "configs/e1_PO_louvain_full_local.yaml"),   # reference: protocol N local training without communication
    ("CE", "configs/e3_CE_central_budget2_p20.yaml"),  # E3: centralized GCN/MLP, 1,500 steps, patience-20 checkpoint recorded
    ("CH", "configs/e3_CH_central_budget2_p100.yaml"), # E3: same, patience 100 steps (= 20 rounds of 5 local steps)
    ("FN", "configs/e3_FN_fedavg_full_predictions.yaml"),  # E3: FedAvg GCN/MLP, protocol N rerun with predictions
] + [(f"PQ{k}", f"configs/e1_PQ{k}_fixed{k}_full.yaml") for k in range(5)] \
  + [(f"PQ{k}", f"configs/e1_PQ{k}_fixed{k + 37}_full.yaml") for k in range(5, 10)]
# protocol Q: the paper's fixed partitions C0-C9, full budget
# C5-C9 partition seeds 42-46 (manuscript Sec. 3.3)
# mistake; they are kept in results/_superseded_PQ5-9_seeds5-9 and are not analysed
LOCAL_FEATURE_EXPERIMENTS = set()
JOBS = {}
for exp, cfg in EXPERIMENTS:
    for i, s in enumerate(SEED_BLOCKS, 1):
        JOBS[f"{exp}{i}"] = {"experiment": exp, "config": cfg, "seeds": s}
SMOKE = {"PSMOKE": {"experiment": "PSMOKE", "config": "configs/e1_PA_louvain_budget1.yaml", "seeds": "42"}}
JOBS.update(SMOKE)
PRIORITY = [j for j in JOBS if j != "PSMOKE"]

if __name__ == "__main__":
    for k, v in JOBS.items():
        print(k, v)
