# E3 controls

## Reproduction checks (should be 0 or below 1e-6)

| check                                                    |   seeds |   max abs difference |
|:---------------------------------------------------------|--------:|---------------------:|
| GCN: centralized full (CE) = paper budget 2              |      20 |              0       |
| GCN: centralized full (CH) = CE                          |      20 |              0       |
| GCN: FedAvg full (FN) = paper N                          |      20 |              0       |
| GCN: FedAvg patience-20 checkpoint (FN) = paper budget 2 |      20 |              0       |
| MLP: centralized full (CE) = paper budget 2              |      20 |              0.0478  |
| MLP: centralized full (CH) = CE                          |      20 |              0       |
| MLP: FedAvg full (FN) = paper N                          |      20 |              0       |
| MLP: FedAvg patience-20 checkpoint (FN) = paper budget 2 |      20 |              0.00613 |

## Federated-centralized gap under symmetric early stopping (budget 2 trajectories)

| encoder   | rule                                                                |   n | centralized AP   |   FedAvg AP | gap (centralized - FedAvg) [95% CI]   | centralized ahead   |      p |
|:----------|:--------------------------------------------------------------------|----:|:-----------------|------------:|:--------------------------------------|:--------------------|-------:|
| GCN       | paper: federated stops (patience 20 rounds), centralized does not   |  20 | 0.5704           |      0.5356 | +0.0349 [+0.0204, +0.0493]            | 17/20               | 0.0002 |
| GCN       | both stop, patience 20 (rounds / steps)                             |  20 | 0.4949           |      0.5356 | -0.0407 [-0.0757, -0.0056]            | 5/20                | 0.0014 |
| GCN       | both stop, patience 20 rounds / 100 steps (same local steps)        |  20 | 0.5482           |      0.5356 | +0.0126 [-0.0038, +0.0290]            | 12/20               | 0.1429 |
| GCN       | neither stops (full budget)                                         |  20 | 0.5704           |      0.5797 | -0.0093 [-0.0186, +0.0000]            | 7/20                | 0.0583 |
| GCN       | centralized: rule fires (patience 20 / 100 steps), median stop step |  20 | 20 / 20 runs     |             | median 80 / 455                       |                     |        |
| MLP       | paper: federated stops (patience 20 rounds), centralized does not   |  20 | 0.5631           |      0.5312 | +0.0319 [+0.0143, +0.0494]            | 18/20               | 0.0006 |
| MLP       | both stop, patience 20 (rounds / steps)                             |  20 | 0.5068           |      0.5312 | -0.0244 [-0.0354, -0.0134]            | 1/20                | 0.0002 |
| MLP       | both stop, patience 20 rounds / 100 steps (same local steps)        |  20 | 0.5328           |      0.5312 | +0.0016 [-0.0180, +0.0211]            | 7/20                | 0.5459 |
| MLP       | neither stops (full budget)                                         |  20 | 0.5631           |      0.5485 | +0.0146 [+0.0037, +0.0254]            | 14/20               | 0.0056 |
| MLP       | centralized: rule fires (patience 20 / 100 steps), median stop step |  20 | 20 / 20 runs     |             | median 52 / 134                       |                     |        |

## Per-client inference: pooled AP, mean of client APs, positive-weighted mean

| protocol   | metric       | design           | vs              |   n |   design AP |   reference AP | difference [95% CI]        | design ahead   | Holm p   |
|:-----------|:-------------|:-----------------|:----------------|----:|------------:|---------------:|:---------------------------|:---------------|:---------|
| A          | pooled       | FedPub_GCN       | FedAvg_GCN      |  20 |      0.4358 |         0.5204 | -0.0846 [-0.1082, -0.0609] | 1/20           | <0.0001  |
| A          | pooled       | FedPub_GCN       | FedGNN_Soft_GCN |  20 |      0.4358 |         0.5128 | -0.0770 [-0.1023, -0.0517] | 1/20           | <0.0001  |
| A          | pooled       | FedGNN_Soft_GCN  | FedAvg_GCN      |  20 |      0.5128 |         0.5204 | -0.0076 [-0.0174, +0.0022] | 7/20           | 0.1054   |
| A          | client_mean  | FedPub_GCN       | FedAvg_GCN      |  20 |      0.5303 |         0.5728 | -0.0425 [-0.0579, -0.0272] | 2/20           | 0.0002   |
| A          | client_mean  | FedPub_GCN       | FedGNN_Soft_GCN |  20 |      0.5303 |         0.5636 | -0.0333 [-0.0499, -0.0168] | 3/20           | 0.0010   |
| A          | client_mean  | FedGNN_Soft_GCN  | FedAvg_GCN      |  20 |      0.5636 |         0.5728 | -0.0092 [-0.0183, -0.0001] | 5/20           | 0.0400   |
| A          | client_wmean | FedPub_GCN       | FedAvg_GCN      |  20 |      0.4975 |         0.5426 | -0.0451 [-0.0604, -0.0298] | 2/20           | 0.0002   |
| A          | client_wmean | FedPub_GCN       | FedGNN_Soft_GCN |  20 |      0.4975 |         0.5345 | -0.0370 [-0.0547, -0.0194] | 3/20           | 0.0012   |
| A          | client_wmean | FedGNN_Soft_GCN  | FedAvg_GCN      |  20 |      0.5345 |         0.5426 | -0.0081 [-0.0175, +0.0013] | 5/20           | 0.0759   |
| N          | pooled       | FedPub_GCN       | FedAvg_GCN      |  20 |      0.4432 |         0.5797 | -0.1365 [-0.1631, -0.1098] | 0/20           | <0.0001  |
| N          | pooled       | FedPub_GCN_local | FedAvg_GCN      |  20 |      0.3761 |         0.5797 | -0.2036 [-0.2241, -0.1831] | 0/20           | <0.0001  |
| N          | pooled       | FedPub_GCN_lr1e3 | FedAvg_GCN      |  20 |      0.4845 |         0.5797 | -0.0952 [-0.1146, -0.0758] | 0/20           | <0.0001  |
| N          | pooled       | FedPub_GCN_wloss | FedAvg_GCN      |  20 |      0.4638 |         0.5797 | -0.1159 [-0.1426, -0.0893] | 0/20           | <0.0001  |
| N          | client_mean  | FedPub_GCN       | FedAvg_GCN      |  20 |      0.5546 |         0.6152 | -0.0606 [-0.0738, -0.0473] | 0/20           | <0.0001  |
| N          | client_mean  | FedPub_GCN_local | FedAvg_GCN      |  20 |      0.4934 |         0.6152 | -0.1217 [-0.1334, -0.1100] | 0/20           | <0.0001  |
| N          | client_mean  | FedPub_GCN_lr1e3 | FedAvg_GCN      |  20 |      0.5766 |         0.6152 | -0.0385 [-0.0494, -0.0277] | 1/20           | <0.0001  |
| N          | client_mean  | FedPub_GCN_wloss | FedAvg_GCN      |  20 |      0.5601 |         0.6152 | -0.0551 [-0.0678, -0.0424] | 1/20           | <0.0001  |
| N          | client_wmean | FedPub_GCN       | FedAvg_GCN      |  20 |      0.5239 |         0.5909 | -0.0670 [-0.0807, -0.0534] | 0/20           | <0.0001  |
| N          | client_wmean | FedPub_GCN_local | FedAvg_GCN      |  20 |      0.4606 |         0.5909 | -0.1303 [-0.1435, -0.1171] | 0/20           | <0.0001  |
| N          | client_wmean | FedPub_GCN_lr1e3 | FedAvg_GCN      |  20 |      0.5503 |         0.5909 | -0.0406 [-0.0521, -0.0292] | 1/20           | <0.0001  |
| N          | client_wmean | FedPub_GCN_wloss | FedAvg_GCN      |  20 |      0.5287 |         0.5909 | -0.0622 [-0.0747, -0.0497] | 0/20           | <0.0001  |

Runs analysed: 160; problems: 0
