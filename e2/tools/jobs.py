"""AMLworld (E2) job table

    python tools/jobs.py   # print the table
"""

SEED_BLOCKS = ["42-46", "47-51", "52-56", "57-61"]
EXPERIMENTS = [
    ("AB", "configs/aml_AB_bank_budget1.yaml"),      # bank partition, budget 1, all designs
    ("AL", "configs/aml_AL_louvain_budget1.yaml"),   # louvain partition, budget 1
    ("AC", "configs/aml_AC_country_budget1.yaml"),   # country partition budget 1
    ("AR", "configs/aml_AR_random_budget1.yaml"),    # random partition, budget 1
    ("NB", "configs/aml_NB_bank_full.yaml"),         # bank partition, 300 rounds x 5 epochs, no early stopping
    ("NC", "configs/aml_NC_central_full.yaml"),      # centralized models, full budget
] + [(f"F{k}", f"configs/aml_F{k}_louvain_fixed{k}_budget1.yaml") for k in range(5)]  # fixed Louvain partitions FedAvg vs Soft

LOCAL_FEATURE_EXPERIMENTS = set()
JOBS = {}
for exp, cfg in EXPERIMENTS:
    for i, s in enumerate(SEED_BLOCKS, 1):
        JOBS[f"{exp}{i}"] = {"experiment": exp, "config": cfg, "seeds": s}
SMOKE = {"SMOKE": {"experiment": "SMOKE", "config": "configs/aml_SMOKE.yaml", "seeds": "42"}}
JOBS.update(SMOKE)
PRIORITY = [j for j in JOBS if j != "SMOKE"]

if __name__ == "__main__":
    for k, v in JOBS.items():
        print(k, v)
