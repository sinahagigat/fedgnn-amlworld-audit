# E1 FED-PUB analysis

FED-PUB runs: 300. Metric: per-client test AP. FED-PUB code hashes: 8ae48e32d382, b05918e6059c, c6d18f2f2b32 (later hashes only add the weighted-loss, learning-rate and local-training options; the official path is unchanged).

## Completed runs per design and protocol

| variant                 |   A |   N |   Q0 |   Q1 |   Q2 |   Q3 |   Q4 |   Q5 |   Q6 |   Q7 |   Q8 |   Q9 |
|:------------------------|----:|----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| FedAvg_GCN              |  20 |  20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |
| FedAvg_MLP              |  20 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedGNN_Hard_GCN         |  20 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedGNN_Soft_GCN         |  20 |  20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |
| FedGNN_Soft_Virtual_GAT |  20 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedGNN_Soft_Virtual_GCN |  20 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedProto_GCN            |  20 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedProx_GCN             |  20 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedPub_GCN              |  20 |  20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |   20 |
| FedPub_GCN_local        |   0 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedPub_GCN_lr1e3        |   0 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
| FedPub_GCN_wloss        |   0 |  20 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |

## Partition identity check: FED-PUB clients equal the paper's clients (same seed)

| protocol   | variant                                 |   seeds |   checked |   identical |
|:-----------|:----------------------------------------|--------:|----------:|------------:|
| A          | FED-PUB                                 |      20 |        20 |          20 |
| N          | FED-PUB                                 |      20 |        20 |          20 |
| N          | FED-PUB (weighted loss)                 |      20 |        20 |          20 |
| N          | FED-PUB (lr 1e-3)                       |      20 |        20 |          20 |
| N          | Local (FED-PUB model, no communication) |      20 |        20 |          20 |
| Q0         | FED-PUB                                 |      20 |        20 |          20 |
| Q1         | FED-PUB                                 |      20 |        20 |          20 |
| Q2         | FED-PUB                                 |      20 |        20 |          20 |
| Q3         | FED-PUB                                 |      20 |        20 |          20 |
| Q4         | FED-PUB                                 |      20 |        20 |          20 |
| Q5         | FED-PUB                                 |      20 |        20 |          20 |
| Q6         | FED-PUB                                 |      20 |        20 |          20 |
| Q7         | FED-PUB                                 |      20 |        20 |          20 |
| Q8         | FED-PUB                                 |      20 |        20 |          20 |
| Q9         | FED-PUB                                 |      20 |        20 |          20 |

## FED-PUB against FedAvg and Soft, per-client test AP (Holm over the two protocols)

| protocol                                           |   n |   FedAvg |   Soft |   FED-PUB | FED-PUB minus FedAvg [95% CI]   |   FED-PUB wins | Holm p (FedAvg)   | FED-PUB minus Soft [95% CI]   | Holm p (Soft)   |
|:---------------------------------------------------|----:|---------:|-------:|----------:|:--------------------------------|---------------:|:------------------|:------------------------------|:----------------|
| A: redrawn Louvain, budget 1, early stopping       |  20 |   0.5204 | 0.5128 |    0.4358 | -0.0846 [-0.1082, -0.0609]      |              1 | <0.0001           | -0.0770 [-0.1023, -0.0517]    | <0.0001         |
| N: redrawn Louvain, full budget, no early stopping |  20 |   0.5797 | 0.5385 |    0.4432 | -0.1365 [-0.1631, -0.1098]      |              0 | <0.0001           | -0.0953 [-0.1219, -0.0686]    | <0.0001         |

## Protocol N, per-client test AP: every federated design against FedAvg (Holm over designs)

| design                  |   n |     AP |     SD | minus FedAvg [95% CI]      |   wins | Holm p   |
|:------------------------|----:|-------:|-------:|:---------------------------|-------:|:---------|
| FedAvg                  |  20 | 0.5797 | 0.0152 |                            |        |          |
| FedProx                 |  20 | 0.5803 | 0.0149 | +0.0006 [-0.0010, +0.0022] |     14 | 0.1429   |
| FedProto                |  20 | 0.566  | 0.0186 | -0.0137 [-0.0197, -0.0078] |      5 | 0.0019   |
| Hard                    |  20 | 0.5491 | 0.0258 | -0.0306 [-0.0461, -0.0151] |      3 | 0.0019   |
| Soft                    |  20 | 0.5385 | 0.0464 | -0.0412 [-0.0632, -0.0192] |      3 | 0.0019   |
| Soft-VN                 |  20 | 0.5355 | 0.0428 | -0.0443 [-0.0635, -0.0250] |      3 | 0.0008   |
| Soft-VN-GAT             |  20 | 0.5379 | 0.0294 | -0.0418 [-0.0563, -0.0274] |      0 | <0.0001  |
| FedAvg-MLP              |  20 | 0.5485 | 0.0227 | -0.0312 [-0.0424, -0.0200] |      2 | 0.0003   |
| FED-PUB                 |  20 | 0.4432 | 0.054  | -0.1365 [-0.1631, -0.1098] |      0 | <0.0001  |
| FED-PUB (weighted loss) |  20 | 0.4638 | 0.0557 | -0.1159 [-0.1426, -0.0893] |      0 | <0.0001  |

## Fixed Louvain partitions Q0-Q9, full budget, per-client test AP (Holm over partitions)

| partition   |   n |   FedAvg |   Soft |   FED-PUB | FED-PUB minus FedAvg [95% CI]   |   FED-PUB wins | Holm p   |   FED-PUB minus Soft | Holm p (Soft)   |   Soft minus FedAvg | Holm p (Soft-FedAvg)   |
|:------------|----:|---------:|-------:|----------:|:--------------------------------|---------------:|:---------|---------------------:|:----------------|--------------------:|:-----------------------|
| Q0          |  20 |   0.5785 | 0.5783 |    0.4435 | -0.1350 [-0.1624, -0.1076]      |              0 | <0.0001  |              -0.1348 | <0.0001         |             -0.0002 | 0.9854                 |
| Q1          |  20 |   0.5831 | 0.5032 |    0.4602 | -0.1229 [-0.1371, -0.1087]      |              0 | <0.0001  |              -0.043  | 0.0004          |             -0.0799 | <0.0001                |
| Q2          |  20 |   0.5825 | 0.573  |    0.4511 | -0.1314 [-0.1543, -0.1086]      |              0 | <0.0001  |              -0.1219 | <0.0001         |             -0.0096 | 0.6976                 |
| Q3          |  20 |   0.5694 | 0.5597 |    0.4471 | -0.1223 [-0.1491, -0.0955]      |              0 | <0.0001  |              -0.1126 | <0.0001         |             -0.0098 | 0.2087                 |
| Q4          |  20 |   0.5631 | 0.5197 |    0.43   | -0.1331 [-0.1471, -0.1190]      |              0 | <0.0001  |              -0.0898 | <0.0001         |             -0.0433 | <0.0001                |
| Q5          |  20 |   0.5731 | 0.534  |    0.4577 | -0.1155 [-0.1323, -0.0987]      |              0 | <0.0001  |              -0.0764 | <0.0001         |             -0.0391 | 0.0007                 |
| Q6          |  20 |   0.577  | 0.5477 |    0.4542 | -0.1228 [-0.1404, -0.1052]      |              0 | <0.0001  |              -0.0936 | <0.0001         |             -0.0292 | 0.0007                 |
| Q7          |  20 |   0.5886 | 0.5782 |    0.4771 | -0.1114 [-0.1273, -0.0955]      |              0 | <0.0001  |              -0.1011 | <0.0001         |             -0.0104 | 0.1600                 |
| Q8          |  20 |   0.571  | 0.5158 |    0.473  | -0.0980 [-0.1112, -0.0847]      |              0 | <0.0001  |              -0.0428 | <0.0001         |             -0.0552 | <0.0001                |
| Q9          |  20 |   0.5644 | 0.5534 |    0.4216 | -0.1427 [-0.1572, -0.1282]      |              0 | <0.0001  |              -0.1318 | <0.0001         |             -0.0109 | 0.1332                 |

## Sensitivity: class-weighted loss (PW) against the official loss (PN), protocol N

| comparison                                        |   n |   mean a |   mean b | a minus b [95% CI]         |   wins a | p       |
|:--------------------------------------------------|----:|---------:|---------:|:---------------------------|---------:|:--------|
| weighted minus official, full budget              |  20 |   0.4638 |   0.4432 | +0.0206 [-0.0012, +0.0423] |       13 | 0.0973  |
| weighted minus official, early-stopped checkpoint |  20 |   0.4497 |   0.4439 | +0.0058 [-0.0151, +0.0267] |        8 | 0.9273  |
| weighted FED-PUB minus FedAvg, full budget        |  20 |   0.4638 |   0.5797 | -0.1159 [-0.1426, -0.0893] |        0 | <0.0001 |
| official FED-PUB minus FedAvg, full budget        |  20 |   0.4432 |   0.5797 | -0.1365 [-0.1631, -0.1098] |        0 | <0.0001 |

## Sensitivity: learning rate 1e-3 (PL) and local training without communication (PO), protocol N

| comparison                         |   n |   mean a |   mean b | a minus b [95% CI]         |   wins a | p       |
|:-----------------------------------|----:|---------:|---------:|:---------------------------|---------:|:--------|
| lr 1e-3 minus official lr 0.01     |  20 |   0.4845 |   0.4432 | +0.0413 [+0.0190, +0.0636] |       16 | 0.0012  |
| FED-PUB (lr 1e-3) minus FedAvg     |  20 |   0.4845 |   0.5797 | -0.0952 [-0.1146, -0.0758] |        0 | <0.0001 |
| FED-PUB minus Local (both lr 0.01) |  20 |   0.4432 |   0.3761 | +0.0671 [+0.0375, +0.0967] |       17 | 0.0002  |
| Local minus FedAvg                 |  20 |   0.3761 |   0.5797 | -0.2036 [-0.2241, -0.1831] |        0 | <0.0001 |

## FED-PUB: full-budget against early-stopped checkpoint

| protocol   | design                                  |   n |   AP full |   AP early-stopped | full minus ES [95% CI]     |      p |   best round (median) |   ES stop round (median) |
|:-----------|:----------------------------------------|----:|----------:|-------------------:|:---------------------------|-------:|----------------------:|-------------------------:|
| N          | FED-PUB                                 |  20 |    0.4432 |             0.4439 | -0.0007 [-0.0128, +0.0114] | 0.7537 |                    77 |                       53 |
| N          | FED-PUB (weighted loss)                 |  20 |    0.4638 |             0.4497 | +0.0141 [-0.0001, +0.0282] | 0.0619 |                    90 |                       52 |
| N          | FED-PUB (lr 1e-3)                       |  20 |    0.4845 |             0.4784 | +0.0061 [-0.0066, +0.0188] | 0.6051 |                   156 |                       49 |
| N          | Local (FED-PUB model, no communication) |  20 |    0.3761 |             0.3564 | +0.0197 [-0.0106, +0.0500] | 0.256  |                   123 |                       29 |
| Q0         | FED-PUB                                 |  20 |    0.4435 |             0.4251 | +0.0184 [+0.0054, +0.0315] | 0.0132 |                   130 |                       58 |
| Q1         | FED-PUB                                 |  20 |    0.4602 |             0.4594 | +0.0007 [-0.0029, +0.0043] | 0.6547 |                    28 |                       48 |
| Q2         | FED-PUB                                 |  20 |    0.4511 |             0.4065 | +0.0446 [+0.0144, +0.0748] | 0.003  |                   165 |                       32 |
| Q3         | FED-PUB                                 |  20 |    0.4471 |             0.3963 | +0.0508 [+0.0336, +0.0680] | 0.0005 |                   176 |                       44 |
| Q4         | FED-PUB                                 |  20 |    0.43   |             0.4252 | +0.0047 [-0.0096, +0.0191] | 0.5303 |                    93 |                       56 |
| Q5         | FED-PUB                                 |  20 |    0.4577 |             0.451  | +0.0066 [-0.0030, +0.0162] | 0.2249 |                    29 |                       46 |
| Q6         | FED-PUB                                 |  20 |    0.4542 |             0.4311 | +0.0230 [+0.0064, +0.0397] | 0.0157 |                   113 |                       54 |
| Q7         | FED-PUB                                 |  20 |    0.4771 |             0.4691 | +0.0081 [-0.0038, +0.0199] | 0.1823 |                    72 |                       49 |
| Q8         | FED-PUB                                 |  20 |    0.473  |             0.4477 | +0.0253 [+0.0095, +0.0411] | 0.0035 |                   166 |                       51 |
| Q9         | FED-PUB                                 |  20 |    0.4216 |             0.4034 | +0.0182 [+0.0059, +0.0306] | 0.0071 |                   152 |                       44 |

## Provenance: code and configuration hashes per job

| job   | source   | code_hash    | config_hash   |   runs |
|:------|:---------|:-------------|:--------------|-------:|
| A1    | paper    | de756535ee20 | cb5bafa43bdf  |     40 |
| A2    | paper    | de756535ee20 | cb5bafa43bdf  |     40 |
| A3    | paper    | de756535ee20 | cb5bafa43bdf  |     40 |
| A4    | paper    | de756535ee20 | cb5bafa43bdf  |     40 |
| M1    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M10   | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M2    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M3    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M4    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M5    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M6    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M7    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M8    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| M9    | paper    | 598ac132e99f | b5b28edd6481  |     12 |
| N1    | paper    | 598ac132e99f | 7b1645612895  |     10 |
| N2    | paper    | 598ac132e99f | 7b1645612895  |     10 |
| N3    | paper    | 598ac132e99f | 7b1645612895  |     10 |
| N4    | paper    | 598ac132e99f | 7b1645612895  |     10 |
| PA1   | e1       | b05918e6059c | f19a680143a8  |      5 |
| PA2   | e1       | b05918e6059c | f19a680143a8  |      5 |
| PA3   | e1       | b05918e6059c | f19a680143a8  |      5 |
| PA4   | e1       | b05918e6059c | f19a680143a8  |      5 |
| PL1   | e1       | c6d18f2f2b32 | 5bd063976a81  |      5 |
| PL2   | e1       | c6d18f2f2b32 | 5bd063976a81  |      5 |
| PL3   | e1       | c6d18f2f2b32 | 5bd063976a81  |      5 |
| PL4   | e1       | c6d18f2f2b32 | 5bd063976a81  |      5 |
| PN1   | e1       | b05918e6059c | 020ec2c0d2c9  |      5 |
| PN2   | e1       | b05918e6059c | 020ec2c0d2c9  |      5 |
| PN3   | e1       | b05918e6059c | 020ec2c0d2c9  |      5 |
| PN4   | e1       | b05918e6059c | 020ec2c0d2c9  |      5 |
| PO1   | e1       | c6d18f2f2b32 | cbf4dcac5973  |      5 |
| PO2   | e1       | c6d18f2f2b32 | cbf4dcac5973  |      5 |
| PO3   | e1       | c6d18f2f2b32 | cbf4dcac5973  |      5 |
| PO4   | e1       | c6d18f2f2b32 | cbf4dcac5973  |      5 |
| PQ01  | e1       | b05918e6059c | d7d88a492fd8  |      5 |
| PQ02  | e1       | b05918e6059c | d7d88a492fd8  |      5 |
| PQ03  | e1       | b05918e6059c | d7d88a492fd8  |      5 |
| PQ04  | e1       | b05918e6059c | d7d88a492fd8  |      5 |
| PQ11  | e1       | b05918e6059c | 2cf8a6f1e917  |      5 |
| PQ12  | e1       | b05918e6059c | 2cf8a6f1e917  |      5 |
| PQ13  | e1       | b05918e6059c | 2cf8a6f1e917  |      5 |
| PQ14  | e1       | b05918e6059c | 2cf8a6f1e917  |      5 |
| PQ21  | e1       | b05918e6059c | 09045e994009  |      5 |
| PQ22  | e1       | b05918e6059c | 09045e994009  |      5 |
| PQ23  | e1       | b05918e6059c | 09045e994009  |      5 |
| PQ24  | e1       | b05918e6059c | 09045e994009  |      5 |
| PQ31  | e1       | b05918e6059c | 26f3cb3967c8  |      5 |
| PQ32  | e1       | b05918e6059c | 26f3cb3967c8  |      5 |
| PQ33  | e1       | b05918e6059c | 26f3cb3967c8  |      5 |
| PQ34  | e1       | b05918e6059c | 26f3cb3967c8  |      5 |
| PQ41  | e1       | b05918e6059c | 5eb5b94a4299  |      5 |
| PQ42  | e1       | b05918e6059c | 5eb5b94a4299  |      5 |
| PQ43  | e1       | b05918e6059c | 5eb5b94a4299  |      5 |
| PQ44  | e1       | b05918e6059c | 5eb5b94a4299  |      5 |
| PQ51  | e1       | 8ae48e32d382 | eab8b1f814e8  |      5 |
| PQ52  | e1       | 8ae48e32d382 | eab8b1f814e8  |      5 |
| PQ53  | e1       | 8ae48e32d382 | eab8b1f814e8  |      5 |
| PQ54  | e1       | 8ae48e32d382 | eab8b1f814e8  |      5 |
| PQ61  | e1       | 8ae48e32d382 | 6b49afdfa281  |      5 |
| PQ62  | e1       | 8ae48e32d382 | 6b49afdfa281  |      5 |
| PQ63  | e1       | 8ae48e32d382 | 6b49afdfa281  |      5 |
| PQ64  | e1       | 8ae48e32d382 | 6b49afdfa281  |      5 |
| PQ71  | e1       | 8ae48e32d382 | c8de58729214  |      5 |
| PQ72  | e1       | 8ae48e32d382 | c8de58729214  |      5 |
| PQ73  | e1       | 8ae48e32d382 | c8de58729214  |      5 |
| PQ74  | e1       | 8ae48e32d382 | c8de58729214  |      5 |
| PQ81  | e1       | 8ae48e32d382 | 0baa94c85cbd  |      5 |
| PQ82  | e1       | 8ae48e32d382 | 0baa94c85cbd  |      5 |
| PQ83  | e1       | 8ae48e32d382 | 0baa94c85cbd  |      5 |
| PQ84  | e1       | 8ae48e32d382 | 0baa94c85cbd  |      5 |
| PQ91  | e1       | 8ae48e32d382 | 73cdb469f979  |      5 |
| PQ92  | e1       | 8ae48e32d382 | 73cdb469f979  |      5 |
| PQ93  | e1       | 8ae48e32d382 | 73cdb469f979  |      5 |
| PQ94  | e1       | 8ae48e32d382 | 73cdb469f979  |      5 |
| PW1   | e1       | 8ae48e32d382 | b103d13a9849  |      5 |
| PW2   | e1       | 8ae48e32d382 | b103d13a9849  |      5 |
| PW3   | e1       | 8ae48e32d382 | b103d13a9849  |      5 |
| PW4   | e1       | 8ae48e32d382 | b103d13a9849  |      5 |
| Q0_1  | paper    | 598ac132e99f | dc7066b668e7  |     10 |
| Q0_2  | paper    | 598ac132e99f | dc7066b668e7  |     10 |
| Q0_3  | paper    | 598ac132e99f | dc7066b668e7  |     10 |
| Q0_4  | paper    | 598ac132e99f | dc7066b668e7  |     10 |
| Q1_1  | paper    | 598ac132e99f | c1cd8ad7ad55  |     10 |
| Q1_2  | paper    | 598ac132e99f | c1cd8ad7ad55  |     10 |
| Q1_3  | paper    | 598ac132e99f | c1cd8ad7ad55  |     10 |
| Q1_4  | paper    | 598ac132e99f | c1cd8ad7ad55  |     10 |
| Q2_1  | paper    | 598ac132e99f | c5d8a1def50b  |     10 |
| Q2_2  | paper    | 598ac132e99f | c5d8a1def50b  |     10 |
| Q2_3  | paper    | 598ac132e99f | c5d8a1def50b  |     10 |
| Q2_4  | paper    | 598ac132e99f | c5d8a1def50b  |     10 |
| Q3_1  | paper    | 598ac132e99f | ec01e7c91358  |     10 |
| Q3_2  | paper    | 598ac132e99f | ec01e7c91358  |     10 |
| Q3_3  | paper    | 598ac132e99f | ec01e7c91358  |     10 |
| Q3_4  | paper    | 598ac132e99f | ec01e7c91358  |     10 |
| Q4_1  | paper    | 598ac132e99f | 5ef7f8b085ba  |     10 |
| Q4_2  | paper    | 598ac132e99f | 5ef7f8b085ba  |     10 |
| Q4_3  | paper    | 598ac132e99f | 5ef7f8b085ba  |     10 |
| Q4_4  | paper    | 598ac132e99f | 5ef7f8b085ba  |     10 |
| Q5_1  | paper    | 598ac132e99f | c3fad1f29230  |     10 |
| Q5_2  | paper    | 598ac132e99f | c3fad1f29230  |     10 |
| Q5_3  | paper    | 598ac132e99f | c3fad1f29230  |     10 |
| Q5_4  | paper    | 598ac132e99f | c3fad1f29230  |     10 |
| Q6_1  | paper    | 598ac132e99f | 4fe39e8d4ebb  |     10 |
| Q6_2  | paper    | 598ac132e99f | 4fe39e8d4ebb  |     10 |
| Q6_3  | paper    | 598ac132e99f | 4fe39e8d4ebb  |     10 |
| Q6_4  | paper    | 598ac132e99f | 4fe39e8d4ebb  |     10 |
| Q7_1  | paper    | 598ac132e99f | 3a20af83154f  |     10 |
| Q7_2  | paper    | 598ac132e99f | 3a20af83154f  |     10 |
| Q7_3  | paper    | 598ac132e99f | 3a20af83154f  |     10 |
| Q7_4  | paper    | 598ac132e99f | 3a20af83154f  |     10 |
| Q8_1  | paper    | 598ac132e99f | bcf012dabfff  |     10 |
| Q8_2  | paper    | 598ac132e99f | bcf012dabfff  |     10 |
| Q8_3  | paper    | 598ac132e99f | bcf012dabfff  |     10 |
| Q8_4  | paper    | 598ac132e99f | bcf012dabfff  |     10 |
| Q9_1  | paper    | 598ac132e99f | ba6214499a90  |     10 |
| Q9_2  | paper    | 598ac132e99f | ba6214499a90  |     10 |
| Q9_3  | paper    | 598ac132e99f | ba6214499a90  |     10 |
| Q9_4  | paper    | 598ac132e99f | ba6214499a90  |     10 |

## Key numbers

- partition_check_all_identical: True
- A_fedavg: 0.5204
- A_soft: 0.5128
- A_fedpub: 0.4358
- A_fedpub_minus_fedavg: -0.0846
- A_fedpub_wins: 1.0
- A_fedpub_minus_fedavg_holm: 0.0
- A_fedpub_minus_soft: -0.077
- A_fedpub_minus_soft_holm: 0.0
- N_fedavg: 0.5797
- N_soft: 0.5385
- N_fedpub: 0.4432
- N_fedpub_minus_fedavg: -0.1365
- N_fedpub_wins: 0.0
- N_fedpub_minus_fedavg_holm: 0.0
- N_fedpub_minus_soft: -0.0953
- N_fedpub_minus_soft_holm: 0.0
- Q_partitions: 10.0
- Q_fedpub_min: 0.4216
- Q_fedpub_max: 0.4771
- Q_fedpub_sd_of_means: 0.0173
- Q_fedavg_sd_of_means: 0.0084
- Q_soft_sd_of_means: 0.0271
- Q_deficit_min: 0.098
- Q_deficit_max: 0.1427
- Q_deficit_ratio: 1.4567
- Q_fedpub_total_wins: 0.0
- Q_fedpub_max_holm: 0.0
- Q_rank_fedpub_last_everywhere: True
- sens_weighted_minus_official: 0.0206
- sens_p_weighted_minus_official: 0.0973
- sens_weighted_minus_official_es: 0.0058
- sens_p_weighted_minus_official_es: 0.9273
- sens_weighted_FED-PUB_minus_FedAvg: -0.1159
- sens_p_weighted_FED-PUB_minus_FedAvg: 0.0
- sens_official_FED-PUB_minus_FedAvg: -0.1365
- sens_p_official_FED-PUB_minus_FedAvg: 0.0
- sens_gap_official: 0.1365
- sens_gap_weighted: 0.1159
- sens_gap_share_explained: 0.1506
- sens_val_ap_official: 0.7639
- sens_val_ap_weighted: 0.7718
- N_val_ap_fedavg: 0.8356
- sens2_lr_minus_official: 0.0413
- sens2_p_lr_minus_official: 0.0012
- sens2_wins_lr_minus_official: 16.0
- sens2_lr_minus_fedavg: -0.0952
- sens2_p_lr_minus_fedavg: 0.0
- sens2_wins_lr_minus_fedavg: 0.0
- sens2_official_minus_local: 0.0671
- sens2_p_official_minus_local: 0.0002
- sens2_wins_official_minus_local: 17.0
- sens2_local_minus_fedavg: -0.2036
- sens2_p_local_minus_fedavg: 0.0
- sens2_wins_local_minus_fedavg: 0.0
- sens2_lr_ap: 0.4845
- sens2_lr_val_ap: 0.7792
- sens2_lr_best_round_median: 156.5
- sens2_local_ap: 0.3761
- sens2_local_val_ap: 0.7067
- sens2_local_best_round_median: 123.5
- fedpub_code_hashes: ['8ae48e32d382', 'b05918e6059c', 'c6d18f2f2b32']
- n_fedpub_runs: 300.0