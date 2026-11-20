# UAVSwarm Benchmark Evaluation Report: Pure Tracking (Tier A), Detection (Tier D) & End-to-End MOT (Tier B)

> **Evaluation Scope**: Comprehensive benchmark of `DroneRLTracker` across **all 36 official UAVSwarm test sequences** (`5,754` frames, `325` ground-truth drone tracks, `37,947` MOT ground-truth boxes) under both:
> - **Tier A (Oracle-Detection / Pure Association, Sections 1–6)** via [run2.py](../run2.py) / [evaluate_tracking.py](evaluate_tracking.py); and
> - **Tier D (Object Detection) + Tier B (Full End-to-End Detection + Tracking, Section 7)** using our NVIDIA L4 cloud fine-tuned **`D-FINE HGNetv2-M`** (`640×640`) and **`YOLOX-M`** (`640×640`) detectors via [evaluate_end_to_end.py](evaluate_end_to_end.py).

---

## 1. Executive Summary (Tier A Oracle Tracking & Tier B End-to-End Overview)

| Tier / Benchmark Split | Detector + Tracker Method | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | IDSW | Avg Purity (%) | FPS (Tracker CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier A: All 36 Seqs** (`GT` boxes) | **`DroneRLTracker` (Ours)** | **99.84** | **97.14** | **99.96** | **94.41** | **96.12** | 44 | **99.32** | 1,599.5 |
| **Tier A: All 36 Seqs** (`GT` boxes) | `DynGate_Hungarian_NoRL` (Pure `dist`) | **99.84** | 97.13 | **99.96** | 94.39 | **96.12** | **42** | **99.32** | 3,631.1 |
| **Tier A: All 36 Seqs** (`GT` boxes) | `Kalman_IoU_Hungarian` (SORT-style) | 99.00 | 94.33 | 99.52 | 89.41 | 92.31 | 197 | 99.03 | 3,462.6 |
| **Tier A: All 36 Seqs** (`GT` boxes) | `Static250_GreedyRL` (Shrimp base) | 99.19 | 92.96 | **100.00** | 86.41 | 91.19 | 307 | 91.97 | 1,559.0 |
| **Tier A: Local 4** (`02, 04, 08, 14`) | **`DroneRLTracker` (Ours)** | **99.99** | **99.88** | **100.00** | **99.77** | **99.84** | **1** | **100.00** | 600.0 |
| **Tier A: Subset** (`06, 28, 30, 46`) | **`DroneRLTracker` (Ours)** | 99.58 | **94.09** | **100.00** | **88.53** | **92.96** | 20 | **99.04** | 1,259.7 |
| **Tier B: All 36 Seqs** (`End-to-End`) | **`D-FINE-M (640)` + `DroneRLTracker`** | **82.87** | **68.12** | **64.93** | **72.52** | 86.91 | 96 | **99.43** | 1,735.1 *(+34.8ms L4 det)* |
| **Tier B: All 36 Seqs** (`End-to-End`) | **`YOLOX-M (640)` + `DroneRLTracker`** | 81.94 | 66.85 | 62.84 | **72.52** | **87.18** | **54** | 99.42 | 1,766.8 *(+17.5ms L4 det)* |
| **Tier B: Local 4** (`02, 04, 08, 14`) | **`D-FINE-M (640)` + `DroneRLTracker`** | 90.73 | **68.95** | **66.82** | **71.90** | 94.60 | 11 | **100.00** | 666.3 |
| **Tier B: Local 4** (`02, 04, 08, 14`) | **`YOLOX-M (640)` + `DroneRLTracker`** | **91.32** | 68.62 | 66.53 | 71.42 | **95.22** | **3** | **100.00** | 692.0 |
| **Tier B: Subset** (`06, 28, 30, 46`) | **`D-FINE-M (640)` + `DroneRLTracker`** | **79.52** | **62.53** | **63.73** | 63.44 | **77.68** | 37 | 97.72 | 1,360.4 |
| **Tier B: Subset** (`06, 28, 30, 46`) | **`YOLOX-M (640)` + `DroneRLTracker`** | 59.59 | 60.27 | 54.40 | **68.40** | 73.86 | **19** | **98.80** | 1,435.1 |

### Key Takeaways
1. **36-Sequence Dataset Performance (`DroneRLTracker`)**: Across `37,947` ground-truth drone annotations in `36` unseen test flights, `DroneRLTracker` achieves **`99.84%` MOTA**, **`97.14%` HOTA** (`DetA = 99.96%`, `AssA = 94.41%`, `LocA = 99.98%`), **`96.12%` IDF1**, **`99.32%` Track Purity** (`96.88%` 100%-pure tracks), and **`44` ID switches** at **`1,599.5 FPS`** on CPU—outperforming both the SORT-style IoU baseline (`94.33%` HOTA, `197` IDSW) and the original `Shrimp` static-gate greedy baseline (`92.96%` HOTA, `307` IDSW).
2. **Design Synergies**:
   - **Velocity-Adaptive Dynamic Gating** ($R_{\text{dynamic}} \in [20, 65]\text{ px}$) + **Global Hungarian Matching** reduces ID switches by **`7.0×` (`307 → 44`)** and boosts HOTA by **`+4.18%` (`92.96% → 97.14%`)** over `Static250_GreedyRL`.
   - **Normalized RL Cost Weighting** ($\tilde{Q} = \tanh((Q - 17.5)/2.5) \in (-1, 1)$ in $C_{i,j} = d_{i,j}(1.5 - \tilde{Q})$) maps cumulative $\gamma = 0.95$ DQN returns into a strictly positive distance-modulating multiplier $(0.5, 2.5)$, achieving the highest overall `HOTA` (`97.14%`) and `AssA` (`94.41%`).
3. **23 of 36 Sequences Score 100.00% Across Every Metric**: `31 / 36` sequences achieve **`> 92%` HOTA**, and `0` sequences lose a track permanently (`ML = 0`, `324 MT / 1 PT / 0 ML`).

---

## 2. Evaluation Protocol & Metrics Definitions

### 2.1 Strictly Isolated Per-Frame Detection Setup
To isolate pure tracking performance without detector noise or label leakage (matching [run2.py](../run2.py)):
1. **Source Data**: All 36 official test sequences from `UAVSwarm-dataset/test/UAVSwarm-{02..72}` stored in [data/uavswarm_test_gt/](data/uavswarm_test_gt/). `conf == 0` ignore flags are filtered out.
2. **Zero Temporal/ID Leakage**: Every frame passes **only** `{'cx', 'cy', 'box': [x1, y1, x2, y2], 'score': 1.0}` to the tracker—ground-truth `track_id` is **never** passed.
3. **Deterministic Per-Frame Shuffling**: Before every frame `f_idx`, detections are shuffled via `random.Random(42 + seq_idx * 10000 + f_idx)` so list order conveys zero identity clues and every tracker receives the exact same input order.
4. **Export Rule Parity**: Predictions are exported using `run2.py`'s exact rule (`len(pts) >= 2` per track; a `min_len >= 1` diagnostic is also reported). Evaluated on `torch.device("cpu")` across `1..seqLength`.

### 2.2 Evaluation Metrics Implemented ([mot_metrics.py](mot_metrics.py))
All metric accumulators are computed per sequence and aggregated globally across sequences following TrackEval conventions:
- **MOTA (Multiple Object Tracking Accuracy)** *(Bernardin & Stiefelhagen, 2008; IoU $\ge 0.50$)*:
  $$\text{MOTA} = 1 - \frac{\text{FP} + \text{FN} + \text{IDSW}}{\text{Total GT}}$$
- **IDF1 (Identity F1 Score)** *(Ristani et al., ECCV 2016; IoU $\ge 0.50$)*:
  Global 1-to-1 trajectory matching between GT IDs and Predicted IDs maximizing `IDTP`:
  $$\text{IDP} = \frac{\text{IDTP}}{\text{IDTP} + \text{IDFP}}, \qquad \text{IDR} = \frac{\text{IDTP}}{\text{IDTP} + \text{IDFN}}, \qquad \text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}$$
- **HOTA (Higher Order Tracking Accuracy)** *(Luiten et al., IJCV 2021)*:
  Evaluated across **19 localization IoU thresholds** $\alpha \in \{0.05, 0.10, \dots, 0.95\}$:
  $$\text{DetA}_\alpha = \frac{\text{TP}_\alpha}{\text{TP}_\alpha + \text{FN}_\alpha + \text{FP}_\alpha}, \qquad \text{AssA}_\alpha = \frac{1}{\text{TP}_\alpha} \sum_{c \in \text{TP}_\alpha} \frac{|\text{TPA}(c)|}{|\text{TPA}(c)| + |\text{FNA}(c)| + |\text{FPA}(c)|}, \qquad \text{HOTA} = \frac{1}{19} \sum_{\alpha} \sqrt{\text{DetA}_\alpha \cdot \text{AssA}_\alpha}$$

*(All metric implementations pass exact analytical unit tests in [mot_metrics.py](mot_metrics.py) for perfect tracking, mid-sequence swaps, and 1-frame occlusions.)*

---

## 3. Full 36-Sequence Ablation Study

![36-Sequence Ablation Comparison](results/plots/ablation_comparison.png)

### 3.1 Aggregate Results Across All 36 UAVSwarm Test Sequences (`5,754` frames, `325` GT tracks, `37,947` GT boxes)

| Method / Ablation | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | LocA (%) | IDF1 (%) | IDP (%) | IDR (%) | FP | FN | IDSW | Frag | MT / PT / ML | Avg Purity (%) | 100%-Pure Tracks | Jumps $>25\text{px}$ | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DroneRLTracker` (Ours)** | **99.84** | **97.14** | 99.96 | **94.41** | **99.98** | **96.12** | **96.14** | **96.10** | 0 | 17 | 44 | 2 | 324 / 1 / 0 | **99.32** | **96.88% (`341/352`)** | 77 | 1,599.5 |
| **`DroneRLTracker` (`min_len=1` diag)** | **99.84** | **97.14** | **100.00** | 94.37 | **99.98** | 96.10 | 96.10 | **96.10** | 0 | **0** | 61 | **0** | 325 / 0 / 0 | **99.35** | **97.02% (`358/369`)** | 77 | 1,599.5 |
| **`DynGate_Hungarian_NoRL` (Pure `dist`)** | **99.84** | 97.13 | 99.96 | 94.39 | **99.98** | **96.12** | **96.14** | 96.09 | 0 | 17 | **42** | 2 | 324 / 1 / 0 | **99.32** | **96.88% (`341/352`)** | 77 | 3,631.1 |
| **`Kalman_IoU_Hungarian` (SORT-style)** | 99.00 | 94.33 | 99.52 | 89.41 | **99.98** | 92.31 | 92.53 | 92.08 | 0 | 184 | 197 | 42 | 321 / 4 / 0 | 99.03 | 96.68% (`466/482`) | **68** | 3,462.6 |
| **`Static250_GreedyRL` (Shrimp base)** | 99.19 | 92.96 | **100.00** | 86.41 | **99.98** | 91.19 | 91.19 | 91.19 | 0 | **0** | 307 | **0** | 325 / 0 / 0 | 91.97 | 74.42% (`256/344`) | 346 | 1,559.0 |

### 3.2 Aggregate Results on the 4 Local Benchmark Sequences (`UAVSwarm-02`, `04`, `08`, `14` — `547` frames, `70` GT tracks, `8,993` GT boxes)

| Method / Ablation | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | IDSW | Avg Purity (%) | 100%-Pure Tracks | Jumps $>25\text{px}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DroneRLTracker` (Ours)** | **99.99** | **99.88** | **100.00** | **99.77** | **99.84** | **1** | **100.00** | **100.00% (`71/71`)** | **0** |
| **`DynGate_Hungarian_NoRL` (Pure `dist`)** | **99.99** | **99.88** | **100.00** | **99.77** | **99.84** | **1** | **100.00** | **100.00% (`71/71`)** | **0** |
| **`Kalman_IoU_Hungarian` (SORT-style)** | 99.96 | 98.90 | **100.00** | 97.82 | 98.13 | 4 | **100.00** | **100.00% (`74/74`)** | **0** |
| **`Static250_GreedyRL` (Shrimp base)** | 99.90 | 96.69 | **100.00** | 93.49 | 95.46 | 9 | 96.83 | 91.55% (`65/71`) | 8 |

### 3.3 Scientific Observations from the Ablations
1. **Dynamic Gating + Hungarian is the Dominant Architectural Win**: Comparing `DroneRLTracker` against `Static250_GreedyRL` shows ID switches drop **`7.0×` (`307 → 44`)**, `HOTA` rises **`+4.18%` (`92.96% → 97.14%`)**, `IDF1` rises **`+4.93%` (`91.19% → 96.12%`)**, and track purity rises **`+7.35%` (`91.97% → 99.32%`)**.
2. **Dynamic Centroid Gating Outperforms Classical IoU Matching**: For tiny aerial drones (often $10\text{--}25\text{ px}$ wide), small prediction shifts quickly push box IoU below `0.30`, causing `Kalman_IoU_Hungarian` to suffer `184` FN, `197` IDSW, and `42` fragmentations (`94.33%` HOTA vs `97.14%` HOTA).
3. **RL vs. Pure Distance (`DroneRLTracker` vs. `NoRL`)**: `DroneRLTracker` matches `NoRL` on `MOTA` (`99.84%`) and `IDF1` (`96.12%`) and leads slightly on `HOTA` (`97.14%` vs `97.13%`, `AssA` `94.41%` vs `94.39%`), while incurring `2` extra IDSW (`44` vs `42`) and `2.3×` slower CPU throughput (`1,599.5` vs `3,631.1 FPS`) due to per-frame MLP scoring. Because `DroneQNetwork` was trained at fixed `gate_radius=250.0`, retraining the policy directly inside `get_dynamic_gate()` (`20–65 px`) is a natural next step to amplify RL gains.

---

## 4. Dedicated 4-Sequence Subset Breakdown: `UAVSwarm-06, 28, 30, 46`

This dedicated subset isolates four contrasting scenarios (`550` total frames, `39` unique GT tracks, `4,789` GT boxes): dense low-res (`06`, median spacing `23.35 px`), fast multi-gap (`28`, `5.61 px/f` mean speed, `50.92 px` spacing), clean well-spaced (`30`, `142.15 px` spacing), and dense high-res (`46`, `1279×684`, `16` drones, `53.39 px` spacing).

### 4.1 Aggregate Metrics Across `{UAVSwarm-06, 28, 30, 46}` (`550` frames, `39` GT tracks, `4,789` GT boxes)

| Method | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | LocA (%) | IDF1 (%) | IDP (%) | IDR (%) | FP | FN | IDSW | Frag | MT / PT / ML | Avg Purity (%) | 100%-Pure Tracks | Jumps $>25\text{px}$ | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DroneRLTracker` (Ours)** | 99.58 | **94.09** | **100.00** | **88.53** | **99.96** | **92.96** | **92.96** | **92.96** | 0 | **0** | 20 | **0** | 39 / 0 / 0 | **99.04** | **94.12% (`48/51`)** | 7 | 1,259.7 |
| **`DynGate_Hungarian_NoRL` (Pure `dist`)** | **99.62** | 94.03 | **100.00** | 88.41 | **99.96** | 92.92 | 92.92 | 92.92 | 0 | **0** | **18** | **0** | 39 / 0 / 0 | 99.02 | **94.12% (`48/51`)** | 7 | 2,775.2 |
| **`Kalman_IoU_Hungarian` (SORT-style)** | 94.93 | 84.29 | 97.45 | 72.90 | 99.92 | 80.03 | 81.08 | 79.01 | 0 | 122 | 121 | 30 | 37 / 2 / 0 | 96.72 | 89.92% (`116/129`) | **2** | 2,880.9 |
| **`Static250_GreedyRL` (Shrimp base)** | 98.23 | 82.80 | **100.00** | 68.56 | 99.90 | 78.91 | 78.91 | 78.91 | 0 | **0** | 85 | **0** | 39 / 0 / 0 | 83.19 | 53.06% (`26/49`) | 70 | 1,274.5 |

### 4.2 Per-Sequence Side-by-Side Comparison (`UAVSwarm-06`, `28`, `30`, `46`)

| Sequence (Res / Frames / GT / NN Dist) | Method | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | IDSW | FN | Avg Purity (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UAVSwarm-06** (`446×270`, `76`f, `760` GT, `23.35 px` NN) | *All 4 Methods* | **100.00** | **100.00** | 100.00 | **100.00** | **100.00** | **0** | **0** | **100.00** |
| **UAVSwarm-28** (`810×475`, `131`f, `873` GT, `50.92 px` NN) | **`DroneRLTracker` (Ours)** | **99.08** | **85.85** | **100.00** | **73.70** | **83.73** | **8** | **0** | **100.00** |
| | `DynGate_Hungarian_NoRL` | **99.08** | **85.85** | **100.00** | **73.70** | **83.73** | **8** | **0** | **100.00** |
| | `Kalman_IoU_Hungarian` | 73.42 | 40.61 | 86.14 | 19.14 | 34.34 | 111 | 121 | 96.17 |
| | `Static250_GreedyRL` | 92.90 | 55.72 | **100.00** | 31.05 | 44.33 | 62 | **0** | 63.12 |
| **UAVSwarm-30** (`863×472`, `105`f, `486` GT, `142.15 px` NN) | *All 4 Methods* | **100.00** | **100.00** | 100.00 | **100.00** | **100.00** | **0** | **0** | **100.00** |
| **UAVSwarm-46** (`1279×684`, `238`f, `2,670` GT, `53.39 px` NN) | **`DroneRLTracker` (Ours)** | 99.55 | **93.82** | **100.00** | **88.03** | **92.70** | 12 | **0** | **97.55** |
| | `DynGate_Hungarian_NoRL` | **99.63** | 93.71 | **100.00** | 87.81 | 92.62 | **10** | **0** | 97.51 |
| | `Kalman_IoU_Hungarian` | 99.59 | 86.82 | 99.96 | 75.40 | 84.62 | **10** | 1 | 96.76 |
| | `Static250_GreedyRL` | 99.14 | 81.33 | **100.00** | 66.14 | 80.37 | 23 | **0** | 84.63 |

*(Full CSV exports: [subset_06_28_30_46_summary.csv](results/tier_a/subset_06_28_30_46_summary.csv) and [subset_06_28_30_46_per_sequence.csv](results/tier_a/subset_06_28_30_46_per_sequence.csv).)*

---

## 5. Complete 36-Sequence Per-Sequence Breakdown (`DroneRLTracker`)

![Per-Sequence Performance Across All 36 Sequences](results/plots/per_sequence_scores.png)

*(★ marks the 4 sequences present in `results/`.)*

| Sequence | Res | Frames | GT IDs | GT Boxes | Pred Tracks | Median NN (px) | Mean Spd (px/f) | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | IDSW | FN | Avg Purity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UAVSwarm-02 ★** | `812×428` | 156 | 20 | 2,983 | 20 | 79.64 | 0.53 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| **UAVSwarm-04 ★** | `812×428` | 161 | 20 | 3,034 | 20 | 84.50 | 0.87 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-06 | `446×270` | 76 | 10 | 760 | 10 | 23.35 | 0.45 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| **UAVSwarm-08 ★** | `446×270` | 115 | 10 | 1,090 | 11 | 69.39 | 0.72 | 99.91 | 99.04 | 100.00 | 98.09 | 98.72 | 1 | 0 | 100.00 |
| UAVSwarm-10 | `863×467` | 58 | 7 | 406 | 7 | 87.19 | 1.33 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-12 | `812×428` | 116 | 20 | 2,320 | 20 | 67.08 | 1.13 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| **UAVSwarm-14 ★** | `812×428` | 115 | 20 | 1,886 | 20 | 83.45 | 1.74 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-16 | `812×428` | 65 | 15 | 721 | 15 | 84.52 | 1.68 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-18 | `625×291` | 84 | 9 | 647 | 9 | 91.04 | 2.47 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-20 | `720×479` | 62 | 5 | 304 | 5 | 108.34 | 4.43 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-22 | `764×479` | 72 | 6 | 228 | 6 | 121.26 | 11.14 | 99.12 | 99.15 | 99.12 | 99.18 | 99.56 | 0 | 2 | 100.00 |
| UAVSwarm-24 | `844×455` | 72 | 9 | 500 | 9 | 100.43 | 4.81 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-26 | `863×364` | 126 | 15 | 1,488 | 15 | 68.53 | 0.91 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-28 | `810×475` | 131 | 8 | 873 | 16 | 50.92 | 5.61 | 99.08 | 85.85 | 100.00 | 73.70 | 83.73 | 8 | 0 | 100.00 |
| UAVSwarm-30 | `863×472` | 105 | 5 | 486 | 5 | 142.15 | 1.12 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-32 | `863×467` | 76 | 6 | 320 | 6 | 68.13 | 6.77 | 99.38 | 92.43 | 100.00 | 85.52 | 92.19 | 2 | 0 | 93.84 |
| UAVSwarm-34 | `1279×630` | 176 | 3 | 528 | 3 | 398.26 | 1.30 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-36 | `1279×625` | 81 | 4 | 247 | 5 | 366.99 | 4.88 | 99.60 | 99.59 | 100.00 | 99.19 | 99.19 | 1 | 0 | 100.00 |
| UAVSwarm-38 | `1279×606` | 452 | 4 | 1,649 | 5 | 307.73 | 1.85 | 99.94 | 94.37 | 100.00 | 89.05 | 92.18 | 1 | 0 | 100.00 |
| UAVSwarm-40 | `1919×1079` | 448 | 9 | 3,604 | 9 | 155.95 | 3.22 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-42 | `847×424` | 120 | 3 | 347 | 3 | 300.17 | 2.57 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-44 | `1919×870` | 353 | 4 | 1,308 | 8 | 220.43 | 3.05 | 99.69 | 84.91 | 100.00 | 72.10 | 76.38 | 4 | 0 | 100.00 |
| UAVSwarm-46 | `1279×684` | 238 | 16 | 2,670 | 20 | 53.39 | 0.86 | 99.55 | 93.82 | 100.00 | 88.03 | 92.70 | 12 | 0 | 97.55 |
| UAVSwarm-48 | `1279×598` | 221 | 3 | 663 | 3 | 375.61 | 2.61 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-50 | `625×292` | 170 | 12 | 1,868 | 12 | 74.05 | 1.01 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-52 | `625×291` | 157 | 9 | 750 | 11 | 101.03 | 4.22 | 97.47 | 85.25 | 98.00 | 74.16 | 84.58 | 4 | 15 | 93.94 |
| UAVSwarm-54 | `625×292` | 110 | 12 | 665 | 12 | 114.59 | 3.69 | 99.70 | 94.29 | 100.00 | 88.90 | 91.28 | 2 | 0 | 96.72 |
| UAVSwarm-56 | `625×292` | 89 | 6 | 378 | 6 | 108.26 | 1.77 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-58 | `863×465` | 122 | 8 | 745 | 8 | 143.57 | 2.82 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-60 | `846×341` | 68 | 7 | 314 | 7 | 102.02 | 8.70 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-62 | `847×474` | 140 | 6 | 435 | 6 | 141.43 | 5.72 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-64 | `863×378` | 75 | 7 | 396 | 7 | 181.10 | 1.75 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-66 | `863×464` | 73 | 3 | 219 | 3 | 50.23 | 1.03 | **100.00** | **100.00** | 100.00 | 100.00 | **100.00** | 0 | 0 | 100.00 |
| UAVSwarm-68 | `811×426` | 59 | 19 | 976 | 19 | 38.37 | 3.53 | 99.69 | 98.94 | 100.00 | 97.89 | 98.98 | 3 | 0 | 97.52 |
| UAVSwarm-70 | `639×351` | 307 | 3 | 789 | 7 | 164.80 | 0.55 | 99.49 | 81.98 | 100.00 | 67.21 | 73.13 | 4 | 0 | 100.00 |
| UAVSwarm-72 | `639×351` | 705 | 2 | 1,350 | 4 | 160.46 | 0.92 | 99.85 | 83.34 | 100.00 | 69.45 | 80.74 | 2 | 0 | 100.00 |

- **23 of 36 sequences** achieve **100.00%** across `MOTA`, `HOTA`, and `IDF1`.
- **31 of 36 sequences** achieve **>92% HOTA**.

---

## 6. Deep Error & Failure-Mode Analysis

![Failure and Scaling Analysis](results/plots/failure_and_scaling_analysis.png)

Across all `37,947` boxes, the **44 ID switches and 17 false negatives** come from three well-defined sources:

### 1. Long GT Occlusion Gaps Exceeding `MAX_MISSED = 8` *(Primary cause of lowest HOTA/IDF1 sequences)*
Because `MAX_MISSED = 8` deletes a track after 9 missing frames, long dataset occlusions split a drone into two separate track IDs even when `Track Purity = 100%`:
- `UAVSwarm-70` (`HOTA = 81.98%`, `IDF1 = 73.13%`, `4` IDSW): GT tracks vanish for `14`, `29`, `41`, and `43` frames (`5` total gaps).
- `UAVSwarm-72` (`HOTA = 83.34%`, `IDF1 = 80.74%`, `2` IDSW): GT tracks have `22`-frame and `9`-frame gaps.
- `UAVSwarm-38` (`HOTA = 94.37%`, `IDF1 = 92.18%`, `1` IDSW): Drone #2 disappears for `22` frames (`301 → 324`).
- `UAVSwarm-28` (`HOTA = 85.85%`, `IDF1 = 83.73%`, `8` IDSW): Contains `12` GT gaps up to `28` frames long.
- `UAVSwarm-46` (`HOTA = 93.82%`, `IDF1 = 92.70%`, `12` IDSW): Contains `7` GT gaps up to `22` frames long plus dense (`53.39 px`) crossings.

### 2. High-Resolution & High-Speed Gate Overflow (`MIN_GATE = 20 px`, `MAX_GATE = 65 px`)
The gating constants (`20–65 px`) are calibrated in raw pixels for `812×428` video:
- `UAVSwarm-44` (`1919×870`, **zero** GT gaps, `max_speed = 38.28 px/f`): Suffers `4` IDSW (`HOTA = 84.91%`, `IDF1 = 76.38%`) when abrupt 1080p-scale accelerations exceed the tight gate.
- `UAVSwarm-52` (`625×291`, `max_speed = 64.07 px/f`): Drops to `HOTA = 85.25%` (`15` FN, `4` IDSW) when single-frame motion spikes exceed `MAX_GATE = 65 px` and `step_d > dyn_gate + 5.0` rejects re-acquisition (`15` 1-frame spawns filtered out by `len >= 2`).

### 3. Training-vs.-Inference Feature Scale Shift
`DroneQNetwork` was trained at fixed `gate_radius = 250.0`, while inference divides `[dx, dy, vx, vy]` by `dyn_gate ∈ [20, 65]`, shifting input magnitudes $4\times\text{--}12.5\times$ larger at inference time. Retraining directly inside `get_dynamic_gate()` is the cleanest future improvement.

---

## 7. Cloud Fine-Tuned Detection (Tier D) & Full 36-Sequence End-to-End MOT (Tier B)

### 7.1 Zero-Leakage Training & Validation Protocol
To eliminate any train/test contamination before running end-to-end tracking on the official UAVSwarm test set:
1. **`train_85.json`** (`5,805` images, `42,406` GT boxes): First 85% of frames within each of the 36 training sequences.
2. **`val_15.json`** (`1,039` images, `7,255` GT boxes): Trailing 15% of frames within each training sequence — used **exclusively** to choose the best checkpoint (`best_stg2.pth` epoch 14 for D-FINE; epoch 20 for YOLOX-M) and select the operating confidence threshold maximizing F1.
3. **`test_official.json` / `data/uavswarm_test_gt/`** (`5,754` frames, `37,703` COCO boxes / `37,947` MOT GT boxes across all 36 official test sequences `UAVSwarm-02..72`): Evaluated **strictly once** at those locked `val_15` thresholds (`0.55` for D-FINE-M, `0.50` for YOLOX-M).
4. **Cloud Hardware**: Two parallel **NVIDIA L4 (24 GB)** Colab VMs (A100 allocation returned transient HTTP 503 unavailability).

### 7.2 Object Detection Comparison (Tier D — `val_15` & Unseen `5,754`-Frame `test_official`)

| Detector (`640×640`) | Split | `AP@[.5:.95]` (%) | `AP50` (%) | `AP75` (%) | `AP_small` (%) | `AP_medium` (%) | `AP_large` (%) | `AR@100` (%) | L4 GPU Speed | `val_15` Conf / Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`D-FINE HGNetv2-M`** | `val_15` (`1,039` imgs) | 60.65 | 95.69 | 69.10 | 58.58 | **75.80** | **54.90** | 68.45 | — | `conf=0.55` (`F1=94.25%`) |
| **`YOLOX-M`** | `val_15` (`1,039` imgs) | **67.84** | **96.82** | **78.74** | **67.40** | 74.00 | 53.22 | **71.92** | — | `conf=0.50` (`F1=96.62%`) |
| **`D-FINE HGNetv2-M`** | **Official Test (`5,754` imgs)** | **48.94** | **92.86** | **46.14** | **46.47** | **60.84** | **70.19** | **59.94** | `28.77 FPS` (`34.76 ms`) | `78.57 MB` EMA (`cba23409...`) |
| **`YOLOX-M`** | **Official Test (`5,754` imgs)** | 44.84 | 90.28 | 38.12 | 42.07 | 56.51 | 60.39 | 51.50 | **`57.03 FPS`** (`17.53 ms`) | `101.20 MB` ONNX (`c6a9097d...`) |

**Key Tier-D Insight**: While `YOLOX-M` converges higher on within-sequence `val_15` frames (`67.84% AP`), **`D-FINE HGNetv2-M` generalizes substantially better to unseen test sequences (`+4.10%` AP, `+2.58%` AP50, `+8.02%` AP75, `+4.40%` AP_small, `+8.44%` AR@100)**, whereas `YOLOX-M` runs nearly **`2.0×` faster (`57.03 FPS` vs `28.77 FPS`)** with nearly identical ID-switch stability once tracked.

### 7.3 Full 36-Sequence End-to-End MOT Results (Tier B — `5,754` Frames, `37,947` GT Boxes)

![End-to-End Detector x Tracker Comparison](results/plots/end_to_end_comparison.png)

| Detector (`val_15` Thresh) | Tracker Method | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | LocA (%) | IDF1 (%) | Precision (%) | Recall (%) | FP | FN | IDSW | Frag | Avg Purity (%) | CPU FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D-FINE-M` (`conf=0.55`)** | **`DroneRLTracker` (Ours)** | **82.87** | **68.12** | 64.93 | **72.52** | **80.03** | **86.91** | 92.77 | 90.14 | 2,664 | 3,741 | **96** | 627 | **99.43** | 1,735.1 |
| **`D-FINE-M` (`conf=0.55`)** | `DynGate_Hungarian_NoRL` | 82.86 | 68.09 | 64.93 | 72.48 | **80.03** | 86.88 | 92.77 | 90.14 | 2,664 | 3,741 | 99 | 627 | **99.43** | 3,710.0 |
| **`D-FINE-M` (`conf=0.55`)** | `Kalman_IoU_Hungarian` (SORT) | 82.42 | 67.03 | 64.81 | 70.44 | **80.03** | 84.76 | **92.91** | 89.83 | **2,602** | 3,859 | 209 | **624** | 99.09 | **4,225.8** |
| **`D-FINE-M` (`conf=0.55`)** | `Static250_GreedyRL` (Shrimp) | 80.87 | 62.05 | **64.96** | 60.52 | 80.02 | 75.51 | 92.70 | **90.25** | 2,698 | **3,701** | 861 | 628 | 83.61 | 1,706.7 |
| **`YOLOX-M` (`conf=0.50`)** | **`DroneRLTracker` (Ours)** | **81.94** | **66.85** | 62.84 | **72.52** | **79.01** | **87.18** | 92.78 | 89.00 | 2,627 | 4,173 | **54** | 407 | **99.42** | 1,766.8 |
| **`YOLOX-M` (`conf=0.50`)** | `DynGate_Hungarian_NoRL` | **81.94** | **66.85** | 62.84 | **72.52** | **79.01** | **87.18** | 92.78 | 89.00 | 2,627 | 4,173 | **54** | 407 | **99.42** | 3,296.1 |
| **`YOLOX-M` (`conf=0.50`)** | `Kalman_IoU_Hungarian` (SORT) | 81.37 | 65.65 | 62.65 | 70.23 | **79.01** | 84.61 | **92.84** | 88.65 | **2,593** | 4,307 | 170 | 418 | 99.18 | **3,873.7** |
| **`YOLOX-M` (`conf=0.50`)** | `Static250_GreedyRL` (Shrimp) | 80.62 | 60.87 | **62.88** | 60.49 | 79.00 | 77.21 | 92.76 | **89.07** | 2,637 | **4,147** | 569 | **405** | 88.14 | 1,586.3 |

### 7.4 End-to-End Local 4 (`02, 04, 08, 14`) & Subset `{06, 28, 30, 46}` Tables

#### Aggregate on Local 4 Sequences (`02, 04, 08, 14`) & Subset `{06, 28, 30, 46}`

| Split | Detector | Method | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | FP | FN | IDSW | Avg Purity (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local 4** | `YOLOX-M (0.50)` | **`DroneRLTracker` (Ours)** | **91.32** | 68.62 | 66.53 | 71.42 | **95.22** | 352 | **426** | **3** | **100.00** |
| **Local 4** | `YOLOX-M (0.50)` | `Kalman_IoU_Hungarian` | **91.33** | 68.62 | 66.54 | 71.42 | **95.23** | **351** | **426** | **3** | **100.00** |
| **Local 4** | `YOLOX-M (0.50)` | `Static250_GreedyRL` | 88.58 | 57.27 | 66.53 | 50.20 | 75.35 | 352 | **426** | 249 | 84.18 |
| **Local 4** | `D-FINE-M (0.55)` | **`DroneRLTracker` (Ours)** | 90.73 | 68.95 | 66.82 | 71.90 | 94.60 | 297 | 526 | 11 | **100.00** |
| **Local 4** | `D-FINE-M (0.55)` | `Kalman_IoU_Hungarian` | 90.83 | **69.25** | **66.85** | **72.49** | **95.25** | **292** | 526 | 7 | **100.00** |
| **Local 4** | `D-FINE-M (0.55)` | `Static250_GreedyRL` | 86.96 | 54.63 | 66.78 | 45.63 | 68.25 | 300 | 528 | 345 | 76.96 |
| **Subset `{06,28,30,46}`** | `D-FINE-M (0.55)` | **`DroneRLTracker` (Ours)** | **79.52** | **62.53** | **63.73** | 63.44 | **77.68** | 315 | **629** | 37 | 97.72 |
| **Subset `{06,28,30,46}`** | `D-FINE-M (0.55)` | `DynGate_Hungarian_NoRL` | 79.43 | **62.55** | **63.73** | 63.48 | **77.74** | 315 | **629** | 41 | 97.72 |
| **Subset `{06,28,30,46}`** | `D-FINE-M (0.55)` | `Kalman_IoU_Hungarian` | 76.13 | 58.39 | 62.37 | 56.74 | 69.43 | **297** | 726 | 120 | 97.33 |
| **Subset `{06,28,30,46}`** | `D-FINE-M (0.55)` | `Static250_GreedyRL` | 75.65 | 53.02 | 63.59 | 46.70 | 56.69 | 326 | **629** | 211 | 77.43 |
| **Subset `{06,28,30,46}`** | `YOLOX-M (0.50)` | **`DroneRLTracker` (Ours)** | 59.59 | 60.27 | 54.40 | **68.40** | 73.86 | 699 | 1,217 | **19** | **98.80** |
| **Subset `{06,28,30,46}`** | `YOLOX-M (0.50)` | `Kalman_IoU_Hungarian` | 56.19 | 56.67 | 53.11 | 62.09 | 66.20 | 684 | 1,319 | 95 | 96.94 |
| **Subset `{06,28,30,46}`** | `YOLOX-M (0.50)` | `Static250_GreedyRL` | 58.15 | 54.66 | 54.35 | 56.87 | 64.23 | 701 | 1,218 | 85 | 85.91 |

#### Per-Sequence Tier-B Comparison on `{UAVSwarm-06, 28, 30, 46}` (`DroneRLTracker (Ours)`)

| Sequence | Detector | MOTA (%) | HOTA (%) | DetA (%) | AssA (%) | IDF1 (%) | TP | FP | FN | IDSW | Avg Purity (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`UAVSwarm-06`** | `YOLOX-M (0.50)` | **100.00** | **84.45** | **82.50** | **87.08** | **100.00** | **760** | **0** | **0** | **0** | **100.00** |
| **`UAVSwarm-06`** | `D-FINE-M (0.55)` | 93.29 | 82.44 | 80.46 | 84.85 | 94.44 | 714 | **0** | 46 | 5 | 98.32 |
| **`UAVSwarm-28`** | `D-FINE-M (0.55)` | **71.02** | **56.31** | **60.22** | 52.92 | **74.24** | **698** | 67 | **175** | 11 | 98.24 |
| **`UAVSwarm-28`** | `YOLOX-M (0.50)` | 63.57 | 52.22 | 51.39 | **53.35** | 70.40 | 602 | **42** | 271 | **5** | **100.00** |
| **`UAVSwarm-30`** | `YOLOX-M (0.50)` | **91.56** | **75.54** | **74.14** | 77.30 | **92.39** | **453** | 7 | **33** | **1** | **100.00** |
| **`UAVSwarm-30`** | `D-FINE-M (0.55)` | 88.68 | 73.87 | 70.69 | **77.84** | 91.34 | 435 | **3** | 51 | **1** | **100.00** |
| **`UAVSwarm-46`** | `D-FINE-M (0.55)` | **76.70** | **55.31** | **60.46** | 51.62 | **71.61** | **2,313** | **245** | **357** | 20 | 96.67 |
| **`UAVSwarm-46`** | `YOLOX-M (0.50)` | 40.97 | 51.62 | 45.99 | **59.28** | 63.62 | 1,757 | 650 | 913 | **13** | **97.15** |

### 7.5 Per-Sequence Tier-B Distribution & Diagnostic Confidence Sensitivity

![End-to-End Per-Sequence HOTA Across 36 Unseen Videos](results/plots/end_to_end_per_sequence.png)

- **Sequence-Level Spread (`DroneRLTracker`)**:
  - `YOLOX-M (conf=0.50)`: **`6/36`** sequences achieve `100.00%` MOTA & IDF1 (`UAVSwarm-06, 10, 12, 34, 36, 48`, peaking at `89.07%` HOTA on `UAVSwarm-48`); **`25/36`** achieve `>= 80%` MOTA; **`20/36`** achieve `>= 90%` MOTA; median `HOTA = 68.07%`. Remaining hard outliers drive most detector errors (`UAVSwarm-18`: `-55.49%` MOTA, `359` FP / `647` FN; `UAVSwarm-56`: `-29.10%` MOTA, `385` FP / `98` FN; `UAVSwarm-22`: `0.00%` MOTA, `228` FN).
  - `D-FINE-M (conf=0.55)`: **`3/36`** sequences hit `100.00%` MOTA & IDF1 (`UAVSwarm-10, 34, 48`, peaking at `88.36%` HOTA on `UAVSwarm-48`); **`24/36`** achieve `>= 80%` MOTA; **`17/36`** achieve `>= 90%` MOTA; median **`HOTA = 69.42%`**. Raises `UAVSwarm-46` (`+35.73%` MOTA vs YOLOX-M) and `UAVSwarm-28` (`+7.45%` MOTA), while `UAVSwarm-18` (`-58.42%` MOTA, `379` FP / `646` FN) and `UAVSwarm-22` (`3.51%` MOTA, `17.07%` HOTA) remain challenging unseen scenes.
- **Post-Hoc Confidence Threshold Sensitivity (`DroneRLTracker` Diagnostic — Operating Thresholds `0.50` / `0.55` Locked Prior via `val_15`)**:

| Conf Thresh | `YOLOX-M` MOTA (%) | `YOLOX-M` HOTA (%) | `YOLOX-M` IDF1 (%) | `YOLOX-M` IDSW | `D-FINE-M` MOTA (%) | `D-FINE-M` HOTA (%) | `D-FINE-M` IDF1 (%) | `D-FINE-M` IDSW |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.15` | 80.34 | 66.51 | 86.30 | 55 | 50.63 | 57.52 | 69.19 | 449 |
| `0.25` | 81.30 | 66.65 | 86.61 | 55 | 73.39 | 64.45 | 80.13 | 246 |
| `0.35` | 81.66 | 66.73 | 86.85 | 55 | 79.51 | 67.05 | 84.71 | 163 |
| `0.45` | 81.91 | **66.87** | 87.16 | **53** | 81.94 | 67.86 | 86.21 | 121 |
| **`0.50`** *(YOLOX `val_15`)* | **81.94** | 66.85 | **87.18** | 54 | 82.56 | **68.28** | **87.16** | 104 |
| **`0.55`** *(D-FINE `val_15`)* | 81.86 | 66.37 | 86.06 | 59 | 82.87 | 68.12 | 86.91 | 96 |
| `0.60` | 81.81 | 66.28 | 86.03 | 61 | **82.89** | 67.94 | 86.94 | **94** |

---

## 8. Cost-Matrix Formulation Ablation Study (`testing_different_stratergies/`)

To validate whether simpler or unnormalized Q-value cost mappings inside the velocity-adaptive dynamic gate (`dist <= dyn_gate`) outperform our primary Hungarian cost formulation `dist * (1.5 - tanh((Q - 17.5)/2.5))`, we benchmarked five alternative Hungarian cost strategies across all **36 official test sequences (`5,754` frames, `37,947` GT boxes)** on both **Tier A (`Oracle GT`)** and **Tier B (`D-FINE-M 0.55` & `YOLOX-M 0.50`)** in [testing_different_stratergies/](testing_different_stratergies/).

### 8.1 All-36-Sequence Comparison Across Cost Formulations

| Mode / Detector | Cost Matrix (`dist <= dyn_gate`) | Method Identifier | MOTA (%) ↑ | HOTA (%) ↑ | AssA (%) ↑ | IDF1 (%) ↑ | IDSW ↓ | Avg Purity (%) | FPS (CPU) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier A: Oracle GT** | **`dist * (1.5 - q_norm)`** | **`DroneRLTracker` (Ours)** | **99.84** | **97.14** | **94.41** | **96.12** | 44 | **99.32** | 1,599.5 |
| **Tier A: Oracle GT** | `-q / (d + 1.0)` | `DynGate_Hungarian_NegQOverD_Smooth` | **99.84** | **97.14** | **94.41** | **96.12** | 44 | **99.32** | 1,679.0 |
| **Tier A: Oracle GT** | `-q / (d + 1e-3)` | `DynGate_Hungarian_NegQOverD` | **99.84** | 97.13 | 94.39 | **96.12** | **42** | **99.32** | 1,656.5 |
| **Tier A: Oracle GT** | `d / q` | `DynGate_Hungarian_DOverQ` | **99.84** | 97.13 | 94.39 | **96.12** | **42** | **99.32** | 1,627.9 |
| **Tier A: Oracle GT** | `dist` *(pure distance)* | `DynGate_Hungarian_NoRL` | **99.84** | 97.13 | 94.39 | **96.12** | **42** | **99.32** | **3,631.1** |
| **Tier A: Oracle GT** | `-q` | `DynGate_Hungarian_NegQ` | 99.75 | 96.73 | 93.61 | 95.69 | 77 | 98.62 | 1,353.4 |
| **Tier B: D-FINE-M (`0.55`)** | **`dist * (1.5 - q_norm)`** | **`DroneRLTracker` (Ours)** | **82.87** | **68.12** | **72.52** | **86.91** | **96** | **99.43** | 1,735.1 |
| **Tier B: D-FINE-M (`0.55`)** | `-q / (d + 1.0)` | `DynGate_Hungarian_NegQOverD_Smooth` | **82.87** | 68.09 | 72.47 | 86.88 | 97 | **99.43** | 1,724.0 |
| **Tier B: D-FINE-M (`0.55`)** | `-q / (d + 1e-3)` | `DynGate_Hungarian_NegQOverD` | **82.87** | 68.09 | 72.47 | 86.88 | 97 | **99.43** | 1,572.9 |
| **Tier B: D-FINE-M (`0.55`)** | `d / q` | `DynGate_Hungarian_DOverQ` | 82.86 | 68.09 | 72.48 | 86.88 | 99 | **99.43** | 1,652.4 |
| **Tier B: D-FINE-M (`0.55`)** | `dist` *(pure distance)* | `DynGate_Hungarian_NoRL` | 82.86 | 68.09 | 72.48 | 86.88 | 99 | **99.43** | **3,710.0** |
| **Tier B: D-FINE-M (`0.55`)** | `-q` | `DynGate_Hungarian_NegQ` | 82.66 | 67.90 | 72.06 | 86.43 | 167 | 98.75 | 1,660.0 |
| **Tier B: YOLOX-M (`0.50`)** | **`dist * (1.5 - q_norm)`** | **`DroneRLTracker` (Ours)** | **81.94** | **66.85** | **72.52** | **87.18** | 54 | **99.42** | 1,766.8 |
| **Tier B: YOLOX-M (`0.50`)** | `-q / (d + 1.0)` | `DynGate_Hungarian_NegQOverD_Smooth` | **81.94** | 66.80 | 72.41 | 87.04 | **53** | 99.39 | 1,772.9 |
| **Tier B: YOLOX-M (`0.50`)** | `-q / (d + 1e-3)` | `DynGate_Hungarian_NegQOverD` | **81.94** | 66.80 | 72.41 | 87.04 | **53** | 99.39 | 1,778.7 |
| **Tier B: YOLOX-M (`0.50`)** | `d / q` | `DynGate_Hungarian_DOverQ` | **81.94** | **66.85** | **72.52** | **87.18** | 54 | **99.42** | 1,800.2 |
| **Tier B: YOLOX-M (`0.50`)** | `dist` *(pure distance)* | `DynGate_Hungarian_NoRL` | **81.94** | **66.85** | **72.52** | **87.18** | 54 | **99.42** | **3,296.1** |
| **Tier B: YOLOX-M (`0.50`)** | `-q` | `DynGate_Hungarian_NegQ` | 81.86 | 66.76 | 72.34 | 86.83 | 81 | 98.99 | 1,839.3 |

### 8.2 Dedicated `{UAVSwarm-06, 28, 30, 46}` Comparison Across Cost Strategies

| Mode / Detector | `dist * (1.5 - q_norm)` (`Ours`) | `-q / (d + 1.0)` | `-q / (d + 1e-3)` | `d / q` / `dist` | `-q` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier A (`Oracle GT`)** | `99.58% / 94.09% / 92.96% / 20` | `99.58% / 94.09% / 92.96% / 20` | **`99.62% / 94.03% / 92.92% / 18`** | **`99.62% / 94.03% / 92.92% / 18`** | `99.44% / 91.99% / 90.52% / 27` |
| **Tier B (`D-FINE-M 0.55`)** | **`79.52% / 62.53% / 77.68% / 37`** | `79.47% / 62.53% / 77.68% / 39` | `79.47% / 62.53% / 77.68% / 39` | `79.43% / 62.55% / 77.74% / 41` | `78.85% / 62.17% / 75.67% / 67` |
| **Tier B (`YOLOX-M 0.50`)** | **`59.59% / 60.27% / 73.86% / 19`** | `59.59% / 59.90% / 73.07% / 19` | `59.59% / 59.90% / 73.07% / 19` | **`59.59% / 60.27% / 73.86% / 19`** | `59.18% / 59.11% / 70.77% / 39` |

*(Cells report `MOTA / HOTA / IDF1 / IDSW`.)*

### 8.3 Mathematical & Empirical Conclusions
1. **Tight Positive Q-Value Concentration**: Across `37k–39k` gated candidate evaluations per mode, `DroneQNetwork` predicts tightly clustered positive discounted returns (`Q ∈ [16.33, 19.62]`, mean `~19.07`, std `~0.56`).
2. **Why `dist * (1.5 - tanh((Q - 17.5)/2.5))` Is Best Overall**: Expanding the narrow Q-value band into a bounded positive multiplier `(0.5, 2.5)` preserves exact monotonicity with Euclidean distance while amplifying RL discrimination—yielding the top `HOTA` across Tier A (`97.14%`), Tier B `D-FINE-M` (`68.12%`, lowest `96 IDSW`, top `86.91% IDF1`), and Tier B `YOLOX-M` (`66.85%`, `87.18% IDF1`).
3. **Why `d / q` Matches Pure `dist`**: Because `1/Q` varies by only ~`3%` around `~19.07`, `dist / q` barely alters assignment ordering relative to pure `dist` (`0/36` sequences differ on `YOLOX-M`; exact parity with `DynGate_Hungarian_NoRL`).
4. **Why `-q` and Negative Distance-Scaled Forms Degrade**: Using `-q` alone loses sub-gate Euclidean distance precision and raises ID switches by `+50%–75%` (`44 → 77`, `96 → 167`, `54 → 81`). Meanwhile, negative distance-scaled forms (`-d * q` and `-d / q`, evaluated and omitted from retained runners) invert distance preference under cost minimization, picking the farthest candidate inside the gate and jumping to `2,853–6,007` IDSW. Conversely, `-q / (d + eps)` aligns both `-q` and small `d` toward lower cost and stays within `0.00%–0.05%` HOTA of our primary tracker.

---

## 9. Reproducibility & Generated Artifacts

```bash
# 1. Verify exact MOTA / IDF1 / HOTA metric implementations via unit tests
python3 evaluations/mot_metrics.py

# 2. Re-run full 36-sequence Tier-A oracle tracking benchmark + plots
python3 evaluations/evaluate_tracking.py

# 3. Re-run full 36-sequence Tier-B end-to-end (D-FINE-M & YOLOX-M) benchmark + plots
python3 evaluations/evaluate_end_to_end.py

# 4. Re-run Hungarian cost-matrix strategy ablations (-q, -q/(d+eps), d/q)
python3 evaluations/testing_different_stratergies/evaluate_neg_q.py
python3 evaluations/testing_different_stratergies/evaluate_neg_q_over_d.py
python3 evaluations/testing_different_stratergies/evaluate_d_over_q.py
```

### Organized Structure in `evaluations/`
- **Core Runners & Evaluators**: [mot_metrics.py](mot_metrics.py), [evaluate_tracking.py](evaluate_tracking.py) (Tier A), [evaluate_end_to_end.py](evaluate_end_to_end.py) (Tier B), [results.md](results.md) (this report)
- **`data/`** — Benchmark ground truth & raw detections:
  - [data/uavswarm_test_gt/](data/uavswarm_test_gt/) — Official `gt/gt.txt` and `seqinfo.ini` across all 36 test sequences
  - [data/detections_dfine/](data/detections_dfine/) & [data/detections_yolox/](data/detections_yolox/) — Per-sequence unseen test detections across all 36 videos
- **`results/tier_a/`** — Oracle GT tracking outputs:
  - [ablation_summary.csv](results/tier_a/ablation_summary.csv), [per_sequence_metrics.csv](results/tier_a/per_sequence_metrics.csv), [per_sequence_all_methods.csv](results/tier_a/per_sequence_all_methods.csv), [subset_06_28_30_46_summary.csv](results/tier_a/subset_06_28_30_46_summary.csv), [subset_06_28_30_46_per_sequence.csv](results/tier_a/subset_06_28_30_46_per_sequence.csv), [detailed_results.json](results/tier_a/detailed_results.json)
- **`results/tier_b/`** — Detection & end-to-end MOT outputs:
  - [dfine_eval_metrics.json](results/tier_b/dfine_eval_metrics.json), [yolox_eval_metrics.json](results/tier_b/yolox_eval_metrics.json), [end_to_end_summary.csv](results/tier_b/end_to_end_summary.csv), [end_to_end_per_sequence.csv](results/tier_b/end_to_end_per_sequence.csv), [end_to_end_subset_06_28_30_46.csv](results/tier_b/end_to_end_subset_06_28_30_46.csv), [end_to_end_results.json](results/tier_b/end_to_end_results.json)
- **`results/predictions/`** — Standard MOT16 `.txt` trajectory exports:
  - [results/predictions/mot_oracle/](results/predictions/mot_oracle/), [results/predictions/mot_dfine/](results/predictions/mot_dfine/), [results/predictions/mot_yolox/](results/predictions/mot_yolox/)
- **`results/plots/`** — Publication visualizations:
  - [ablation_comparison.png](results/plots/ablation_comparison.png), [per_sequence_scores.png](results/plots/per_sequence_scores.png), [failure_and_scaling_analysis.png](results/plots/failure_and_scaling_analysis.png), [end_to_end_comparison.png](results/plots/end_to_end_comparison.png), [end_to_end_per_sequence.png](results/plots/end_to_end_per_sequence.png)
- **`testing_different_stratergies/`** — Hungarian cost-matrix ablation scripts & outputs:
  - Runners: [evaluate_neg_q.py](testing_different_stratergies/evaluate_neg_q.py), [evaluate_neg_q_over_d.py](testing_different_stratergies/evaluate_neg_q_over_d.py), [evaluate_d_over_q.py](testing_different_stratergies/evaluate_d_over_q.py)
  - Results (`testing_different_stratergies/results/`): [neg_q_summary.csv](testing_different_stratergies/results/neg_q_summary.csv), [neg_q_results.json](testing_different_stratergies/results/neg_q_results.json), [neg_q_over_d_summary.csv](testing_different_stratergies/results/neg_q_over_d_summary.csv), [neg_q_over_d_results.json](testing_different_stratergies/results/neg_q_over_d_results.json), [d_over_q_summary.csv](testing_different_stratergies/results/d_over_q_summary.csv), [d_over_q_results.json](testing_different_stratergies/results/d_over_q_results.json) (plus corresponding `*_per_sequence.csv` and `*_subset_06_28_30_46.csv`)
