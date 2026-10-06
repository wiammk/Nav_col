# Post-review experiment results

All estimates below use the same frozen protocols across the ten training seeds (42–51), with 200 scenarios per protocol and per-robot success/SPL metrics. The dynamic protocol retains cases with no eligible edge closure and evaluates those episodes unchanged. Weighted FedAvg used 20 rounds × 50 episodes/client at K=3, 5, and 10. The shortest-path baseline is deterministic and recomputes a minimum weighted-distance next hop; it has no joint collision planner. Values are mean across seeds; brackets show the 95% t interval half-width across seeds.

## Success and SPL

| K | Regime | FedAvg success | Visit-weighted success | Shortest-path success | FedAvg SPL | Visit-weighted SPL | Shortest-path SPL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Static | 0.8640 [0.0162] | — | 1.0000 [0] | 0.7752 [0.0205] | — | 1.0000 [0] |
| 1 | Edge closure | 0.7205 [0.0098] | — | 1.0000 [0] | 0.6163 [0.0106] | — | 0.9420 [0] |
| 3 | Static | 0.7348 [0.0125] | 0.7293 [0.0188] | 0.7650 [0] | 0.6608 [0.0119] | 0.6447 [0.0168] | 0.7527 [0] |
| 3 | Edge closure | 0.7027 [0.0121] | 0.6905 [0.0211] | 0.7567 [0] | 0.6247 [0.0114] | 0.6037 [0.0187] | 0.7331 [0] |
| 5 | Static | 0.5585 [0.0110] | 0.5472 [0.0144] | 0.5670 [0] | 0.5033 [0.0108] | 0.4790 [0.0120] | 0.5590 [0] |
| 5 | Edge closure | 0.5444 [0.0082] | 0.5296 [0.0127] | 0.5760 [0] | 0.4865 [0.0083] | 0.4600 [0.0099] | 0.5582 [0] |
| 10 | Static | 0.4102 [0.0058] | 0.3984 [0.0071] | 0.3830 [0] | 0.3723 [0.0042] | 0.3523 [0.0064] | 0.3765 [0] |
| 10 | Edge closure | 0.4047 [0.0055] | 0.3906 [0.0080] | 0.3855 [0] | 0.3652 [0.0043] | 0.3434 [0.0074] | 0.3758 [0] |

## Paired comparisons

Differences are method A minus standard FedAvg Q-learning. Tests are exact two-sided sign-permutation tests over the ten paired training seeds. Holm correction is applied separately to the 12 weighted-aggregation tests and 16 shortest-path tests. Reported intervals are 95% paired t intervals.

| Method A | K | Regime | Success difference [95% CI] | p (Holm) | SPL difference [95% CI] | p (Holm) |
|---|---:|---|---:|---:|---:|---:|
| Visit-weighted | 3 | Static | −0.0055 [−0.0247, 0.0137] | 0.551 (0.592) | −0.0161 [−0.0347, 0.0025] | 0.076 (0.305) |
| Visit-weighted | 3 | Edge closure | −0.0122 [−0.0320, 0.0076] | 0.207 (0.592) | −0.0211 [−0.0398, −0.0024] | 0.037 (0.223) |
| Visit-weighted | 5 | Static | −0.0113 [−0.0298, 0.0072] | 0.197 (0.592) | −0.0243 [−0.0411, −0.0075] | 0.018 (0.125) |
| Visit-weighted | 5 | Edge closure | −0.0148 [−0.0301, 0.0005] | 0.061 (0.303) | −0.0265 [−0.0392, −0.0138] | 0.004 (0.039) |
| Visit-weighted | 10 | Static | −0.0117 [−0.0196, −0.0038] | 0.016 (0.125) | −0.0200 [−0.0263, −0.0137] | 0.002 (0.023) |
| Visit-weighted | 10 | Edge closure | −0.0141 [−0.0229, −0.0053] | 0.012 (0.105) | −0.0217 [−0.0287, −0.0147] | 0.002 (0.023) |
| Shortest path | 1 | Static | +0.1360 [+0.1198, +0.1522] | 0.002 (0.031) | +0.2248 [+0.2043, +0.2453] | 0.002 (0.031) |
| Shortest path | 1 | Edge closure | +0.2795 [+0.2697, +0.2893] | 0.002 (0.031) | +0.3257 [+0.3151, +0.3363] | 0.002 (0.031) |
| Shortest path | 3 | Static | +0.0302 [+0.0177, +0.0427] | 0.002 (0.031) | +0.0920 [+0.0801, +0.1039] | 0.002 (0.031) |
| Shortest path | 3 | Edge closure | +0.0540 [+0.0419, +0.0661] | 0.002 (0.031) | +0.1084 [+0.0970, +0.1198] | 0.002 (0.031) |
| Shortest path | 5 | Static | +0.0085 [−0.0025, +0.0195] | 0.123 (0.123) | +0.0557 [+0.0449, +0.0665] | 0.002 (0.031) |
| Shortest path | 5 | Edge closure | +0.0316 [+0.0234, +0.0398] | 0.002 (0.031) | +0.0717 [+0.0634, +0.0800] | 0.002 (0.031) |
| Shortest path | 10 | Static | −0.0272 [−0.0330, −0.0214] | 0.002 (0.031) | +0.0043 [+0.0001, +0.0085] | 0.057 (0.113) |
| Shortest path | 10 | Edge closure | −0.0192 [−0.0247, −0.0137] | 0.002 (0.031) | +0.0107 [+0.0064, +0.0150] | 0.002 (0.031) |

The visit-count weighting produces numerically lower success and SPL at every evaluated K and regime. After Holm correction, its SPL reduction is statistically detectable at K=10 in both regimes and at K=5 under edge closures; success reductions do not remain significant. The K=3 and K=5 static SPL differences also do not remain significant after correction.

The corrected deterministic weighted-shortest-path baseline reaches SPL=1 at K=1 in the static protocol: the reference distance is recomputed after applying static blocked nodes and closed edges. Under dynamic closures, the reference is the distance at reset before the closure, so detours after the event reduce SPL. For K=3 and K=5 the policy improves SPL in both regimes; at K=10 it has lower success than FedAvg in both regimes, while its SPL is higher under edge closures. Because its evaluation is deterministic on fixed scenario protocols, its across-seed paired comparison uses the repeated fixed baseline against the ten FedAvg training-seed means; these are not ten independent shortest-path runs. The baseline is a greedy per-robot weighted shortest-path policy with the simulator's valid-action/occupancy constraints, not an optimal multi-agent path planner.

## Artifacts

- Per-seed models and evaluation outputs: `visitation_weighted/seed_42` through `seed_51`.
- Shortest-path details and summaries: `shortest_path/robots_K`.
- Machine-readable descriptive estimates: `aggregate_review_results.csv`.
- Machine-readable paired tests and Holm correction: `review_paired_tests.csv`.
- Protocol and run manifest: `review_experiment_manifest.json`.
