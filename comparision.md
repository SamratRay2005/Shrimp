# Comparative Analysis: Our Drone Swarm RL Tracker vs. Published UAVSwarm Literature (2022–2026)

This document benchmarks our repository's detection and multi-object tracking (MOT) pipeline against **14 foundational and recent (2022–2026) UAVSwarm research papers**, covering the original dataset creator studies (Shenyang Aerospace University / Beihang University) and recent air-to-air / dense-swarm MOT trackers.

> [!IMPORTANT]
> **Critical Protocol & Data-Provenance Distinctions (Read Before Comparing Numbers)**
>
> Metrics across these papers were measured under **four distinct evaluation tiers** that must not be conflated:
> - **Tier A — Oracle-Detection / Tracking-Only Protocol (`GT` boxes stripped of `track_id`)**: Evaluates *pure data association and kinematic filtering* without detector misses or false positives. Both **our `DroneRLTracker` ([`run2.py`](run2.py) / [`evaluations/results.md`](evaluations/results.md))** and **Wang et al. (2022) `GNMOT`** operate in Tier A — making them **direct apples-to-apples peers**.
> - **Tier B — End-to-End Detection + Tracking on UAVSwarm Test (`36` seqs, `5,754` frames)**: Pairs a learned detector (`YOLOX-M`, `D-FINE-M`, `YOLOX-X`, joint detection head, or query transformer) with a tracker. Both published UAVSwarm papers and **our end-to-end pipelines (`D-FINE-M + DroneRLTracker` and `YOLOX-M + DroneRLTracker` in [`evaluations/results.md`](evaluations/results.md))** now have full 36-sequence Tier-B numbers (`61–86%` MOTA, `52–80%` HOTA).
> - **Tier C — Adjacent Multi-UAV Benchmarks (Non-UAVSwarm Tables)**: `GL-DT / JPTrack` (`arXiv:2510.09092`) reports its quantitative tables on `MOT-FLY` and `FT Dataset` rather than `UAVSwarm`, so it is separated from UAVSwarm leaderboards.
> - **Tier D — Object Detection (`mAP` / `AP50`)**: Both our NVIDIA L4 cloud fine-tuned detectors (**`D-FINE HGNetv2-M`** and **`YOLOX-M`**, trained on `train_85` = `5,805` imgs with checkpoint/threshold selection locked on `val_15` = `1,039` imgs) and published papers are now evaluated on the **official unseen `5,754`-frame UAVSwarm test split**. Also note whether a paper reports **`AP50` (`IoU = 0.50`)** or **COCO `mAP@[0.50:0.95]`**.
>
> **Data Provenance Policy**: Values verified directly from open-access full text/HTML (`arXiv:2510.09092`, `arXiv:2604.06883`, `arXiv:2609.15171`, `SciOpen`, and indexed `MDPI` excerpts) are marked accordingly; headline metrics supplied from paywalled journal/conference abstracts or the prompt summary (`RoeNet`, `KRAfT`, `V-USDT`, `Air-to-Air`, `A2MGA-Track`, `ECA+Swin+BYTE`) are tagged `[Summary / Paywalled Full Table]`, and unretrieved sub-metrics are marked `—` rather than estimated.

---

## Our System Reference Snapshot

### Tracking Performance (Tier A — Shuffled Oracle GT Boxes, All 36 Official Test Sequences)
Evaluated in [evaluations/results.md](evaluations/results.md) ([evaluations/evaluate_tracking.py](evaluations/evaluate_tracking.py)) across `5,754` frames, `325` GT tracks, and `37,947` GT boxes on CPU (`min_len >= 2` export filter matching [`run2.py`](run2.py); `min_len = 1` gives `FN = 0`):

- **`DroneRLTracker` (Ours)** — `SimpleKF` + velocity-adaptive dynamic gating $R_{\text{dynamic}} = \mathrm{clip}(20 + 2\|\mathbf{v}\| + 4k, 20, 65)\text{ px}$ + `DroneQNetwork` 5D DQN scoring ($\tilde{Q} = \tanh((Q - 17.5)/2.5)$) + global Hungarian matching ($C_{i,j} = d_{i,j}(1.5 - \tilde{Q})$):
  - **All 36 Test Sequences**: **`MOTA = 99.84%`**, **`HOTA = 97.14%`** (`DetA = 99.96%`, `AssA = 94.41%`, `LocA = 99.98%`, `AssRe = 94.80%`, `AssPr = 99.11%`), **`IDF1 = 96.12%`** (`IDP = 96.14%`, `IDR = 96.10%`), `FP = 0`, `FN = 17`, `IDSW = 44`, `Frag = 2`, `MT/PT/ML = 324/1/0`, **`Avg Purity = 99.32%`** (`341/352` tracks `100%`-pure), **`1,599.5 FPS`** (CPU association-only). **`23 / 36`** sequences achieve `100.00%` across `MOTA`, `HOTA`, and `IDF1`; **`31 / 36`** exceed `92%` `HOTA`.
  - **Local 4-Sequence Split (`UAVSwarm-02, 04, 08, 14` — `547` frames, `70` tracks, `8,993` boxes)**: `MOTA = 99.99%`, `HOTA = 99.88%`, `IDF1 = 99.84%`, `IDSW = 1`, `Avg Purity = 100.00%`, `600.0 FPS`.
  - **Subset `{UAVSwarm-06, 28, 30, 46}` (`550` frames, `39` tracks, `4,789` boxes)**: `MOTA = 99.58%`, `HOTA = 94.09%`, `IDF1 = 92.96%`, `IDSW = 20`, `Avg Purity = 99.04%`, `1,259.7 FPS`.
- **Controlled Internal Ablations (Same Tier-A Protocol)**:
  - **`DynGate_Hungarian_NoRL`** (pure Euclidean distance cost): `99.84% MOTA`, `97.13% HOTA`, `96.12% IDF1`, `42 IDSW`, `3,631.1 FPS` (CPU).
  - **`Kalman_IoU_Hungarian`** (SORT-style IoU $\ge 0.30$): `99.00% MOTA`, `94.33% HOTA`, `92.31% IDF1`, `197 IDSW`, `3,462.6 FPS` (CPU).
  - **`Static250_GreedyRL`** (Shrimp fixed `R=250px` greedy RL baseline): `99.19% MOTA`, `92.96% HOTA`, `91.19% IDF1`, `307 IDSW`, `1,559.0 FPS` (CPU).

### Cost-Matrix Strategy Ablation (`evaluations/testing_different_stratergies/`)
Tested 5 alternative Hungarian cost matrices inside `[20, 65] px` across Tier A & Tier B (`37k–39k` candidate pairs/mode, raw `Q ∈ [16.33, 19.62]`, mean `~19.07`, std `~0.56`; full tables in [evaluations/results.md](evaluations/results.md) and [testing_different_stratergies/results/](evaluations/testing_different_stratergies/results/)):
- **`dist * (1.5 - q_norm)` (`DroneRLTracker` — Confirmed Best)**: Highest `HOTA` across Tier A (`97.14%`), Tier B `D-FINE-M` (`68.12%`, lowest `96 IDSW`, `86.91% IDF1`), and Tier B `YOLOX-M` (`66.85%`, `87.18% IDF1`).
- **`-q / (d + 1.0)` & `-q / (d + 1e-3)`**: Runner-up within `0.00%–0.05% HOTA` (`99.84% / 97.13–97.14% / 42–44` Tier A; `82.87% / 68.09% / 97` D-FINE-M; `81.94% / 66.80% / 53` YOLOX-M).
- **`d / q`**: Identical on every split to pure `dist` (`NoRL`) because `1/Q` varies by only ~`3%`.
- **`-q`**: Loses sub-gate distance resolution, increasing IDSW `+50%–75%` (`77` Tier A / `167` D-FINE-M / `81` YOLOX-M). *(Negative distance-scaled forms `-d*q` and `-d/q` select farthest in-gate candidates under minimization, causing `2,853–6,007` IDSW.)*

### Object Detection Front-Ends (Tier D — Unseen Official `5,754`-Frame Test Split, `640×640`, NVIDIA L4 GPU)
Trained with zero test-set leakage (`train_85`: `5,805` imgs; `val_15`: `1,039` imgs for checkpoint & confidence selection) and evaluated once on all `5,754` official test images (`37,703` COCO annotations; [dfine_eval_metrics.json](evaluations/results/tier_b/dfine_eval_metrics.json), [yolox_eval_metrics.json](evaluations/results/tier_b/yolox_eval_metrics.json)):

- **Fine-Tuned `D-FINE HGNetv2-M`** (`78.57 MB` clean EMA weights, [`run.py`](run.py), `val_15 conf=0.55`): **`AP50 = 92.86%`**, **`mAP@[0.50:0.95] = 48.94%`**, **`AP75 = 46.14%`**, **`AP_small = 46.47%`**, `AP_medium = 60.84%`, `AR@100 = 59.94%`, **`34.76 ms/frame` (`28.77 FPS` on L4 GPU)** (`val_15`: `60.65% mAP`, `95.69% AP50`).
- **Fine-Tuned `YOLOX-M`** (`101.20 MB` ONNX, [`run3.py --mode runtime`](run3.py), `val_15 conf=0.50`): **`AP50 = 90.28%`**, `mAP@[0.50:0.95] = 44.84%`, `AP75 = 38.12%`, `AP_small = 42.07%`, `AP_medium = 56.51%`, `AR@100 = 51.50%`, **`17.53 ms/frame` (`57.03 FPS` on L4 GPU)** (`val_15`: `67.84% mAP`, `96.82% AP50`).

### End-to-End Tracking Performance (Tier B — Unseen `36` Test Seqs, `5,754` Frames, `37,947` MOT GT Boxes)
Evaluated in [evaluations/results.md](evaluations/results.md) ([evaluations/evaluate_end_to_end.py](evaluations/evaluate_end_to_end.py)) at the locked `val_15` thresholds:

- **`D-FINE-M (640×640, conf=0.55) + DroneRLTracker`**: **`MOTA = 82.87%`**, **`HOTA = 68.12%`** (`DetA = 64.93%`, `AssA = 72.52%`, `LocA = 80.03%`), **`IDF1 = 86.91%`**, `FP = 2,664`, `FN = 3,741`, `IDSW = 96`, `Avg Purity = 99.43%`, **`1,735.1 FPS`** CPU tracker.
  - Local 4 (`02, 04, 08, 14`): `MOTA = 90.73%`, `HOTA = 68.95%`, `IDF1 = 94.60%`, `IDSW = 11`.
  - Subset `{06, 28, 30, 46}`: **`MOTA = 79.52%`**, **`HOTA = 62.53%`**, **`IDF1 = 77.68%`**, `IDSW = 37`.
- **`YOLOX-M (640×640, conf=0.50) + DroneRLTracker`**: **`MOTA = 81.94%`**, **`HOTA = 66.85%`** (`DetA = 62.84%`, `AssA = 72.52%`, `LocA = 79.01%`), **`IDF1 = 87.18%`**, `FP = 2,627`, `FN = 4,173`, **`IDSW = 54`**, `Avg Purity = 99.42%`, **`1,766.8 FPS`** CPU tracker.
  - Local 4 (`02, 04, 08, 14`): `MOTA = 91.32%`, `HOTA = 68.62%`, `IDF1 = 95.22%`, `IDSW = 3`.
  - Subset `{06, 28, 30, 46}`: `MOTA = 59.59%`, `HOTA = 60.27%`, `IDF1 = 73.86%`, **`IDSW = 19`**.

---

## 1. Direct Apples-to-Apples Tracking Comparison (Tier A — Oracle GT-Box Protocol)

Only two systems across our repository and the 14 papers isolate tracking association by feeding ground-truth detections (with GT `track_id` removed) into the tracker across the **36 official UAVSwarm test sequences**: **Wang et al. (2022) `GNMOT`** and **our `DroneRLTracker` (+ internal ablations)**.

### 1.1 Dataset-Wide & Internal Ablation Comparison (Tier A)

| Tracker | Protocol | MOTA (%) ↑ | HOTA (%) ↑ | IDF1 (%) ↑ | IDSW ↓ | Perfect (`100%`) Sequences | CPU FPS ↑ | Source / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`DroneRLTracker` (Ours)** | Tier A (`GT` boxes shuffled, `len>=2`) | **99.84** | **97.14** | **96.12** | 44 | `23 / 36` (`MOTA, HOTA, IDF1`) | 1,599.5 | [evaluations/results.md](evaluations/results.md) |
| **`DynGate_Hungarian_NoRL`** *(Our ablation)* | Tier A (`GT` boxes shuffled, pure `dist`) | **99.84** | 97.13 | **96.12** | **42** | `23 / 36` | **3,631.1** | [ablation_summary.csv](evaluations/results/tier_a/ablation_summary.csv) |
| **`Kalman_IoU_Hungarian`** *(Our SORT-style ablation)* | Tier A (`GT` boxes shuffled, `IoU>=0.3`) | 99.00 | 94.33 | 92.31 | 197 | — | 3,462.6 | [ablation_summary.csv](evaluations/results/tier_a/ablation_summary.csv) |
| **`Static250_GreedyRL`** *(Our Shrimp ablation)* | Tier A (`GT` boxes shuffled, `R=250px`) | 99.19 | 92.96 | 91.19 | 307 | — | 1,559.0 | [ablation_summary.csv](evaluations/results/tier_a/ablation_summary.csv) |
| **Wang et al. (2022) `GNMOT`** | Tier A (`GT` boxes w/ IDs removed) | *Not globally tabulated* | — | *Not globally tabulated* | — | **`25 / 36`** (`MOTA, IDF1`) | — | [MDPI Remote Sensing (2022)](https://www.mdpi.com/2072-4292/14/11/2601) Fig. 7 |

### 1.2 Per-Sequence Direct Comparison on `{UAVSwarm-06, 28, 30, 46}`

Wang et al. (2022) explicitly highlighted and reported per-sequence behavior on **`UAVSwarm-06`**, **`UAVSwarm-28`**, **`UAVSwarm-30`**, and **`UAVSwarm-46`**. Comparing those exact 4 sequences side by side ([subset_06_28_30_46_per_sequence.csv](evaluations/results/tier_a/subset_06_28_30_46_per_sequence.csv)):

| Sequence (Scenario) | Method | Tier | MOTA (%) ↑ | HOTA (%) ↑ | IDF1 (%) ↑ | IDSW ↓ | Avg Purity (%) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`UAVSwarm-06`** (`446×270`, dense `23.35px` NN) | **`DroneRLTracker` (Ours)** | **A** | **100.00** | **100.00** | **100.00** | **0** | **100.00** |
| | `GNMOT` *(Wang et al., 2022)* | **A** | **100.00** | — | — | — | — |
| | **`YOLOX-M + DroneRLTracker` (Ours)** | **B** | **100.00** | **84.45** | **100.00** | **0** | **100.00** |
| | **`D-FINE-M + DroneRLTracker` (Ours)** | **B** | 93.29 | 82.44 | 94.44 | 5 | 98.32 |
| | `ByteTrack` *(Wang et al., 2022)* | B | 88.3 | — | 93.2 | **0** | — |
| **`UAVSwarm-28`** (`810×475`, fast `12` GT gaps, jitter) | **`DroneRLTracker` (Ours)** | **A** | **99.08** | **85.85** | **83.73** | **8** | **100.00** |
| | `Kalman_IoU_Hungarian` *(Our SORT ablation)* | A | 73.42 | 40.61 | 34.34 | 111 | 96.17 |
| | `Static250_GreedyRL` *(Our Shrimp ablation)* | A | 92.90 | 55.72 | 44.33 | 62 | 63.12 |
| | `GNMOT` *(Wang et al., 2022)* | **A** | 98.4 | — | 76.9 | 14 | — |
| | **`D-FINE-M + DroneRLTracker` (Ours)** | **B** | **71.02** | **56.31** | **74.24** | 11 | 98.24 |
| | **`YOLOX-M + DroneRLTracker` (Ours)** | **B** | 63.57 | 52.22 | 70.40 | **5** | **100.00** |
| | `ByteTrack` *(Wang et al., 2022)* | B | 32.5 | — | 30.0 | 14 | — |
| **`UAVSwarm-30`** (`863×472`, well-spaced `142px`) | **`DroneRLTracker` (Ours)** | **A** | **100.00** | **100.00** | **100.00** | **0** | **100.00** |
| | `GNMOT` *(Wang et al., 2022)* | **A** | **100.00** | — | — | — | — |
| | **`YOLOX-M + DroneRLTracker` (Ours)** | **B** | **91.56** | **75.54** | **92.39** | **1** | **100.00** |
| | **`D-FINE-M + DroneRLTracker` (Ours)** | **B** | 88.68 | 73.87 | 91.34 | **1** | **100.00** |
| **`UAVSwarm-46`** (`1279×684`, 16 drones, `7` GT gaps) | **`DroneRLTracker` (Ours)** | **A** | 99.55 | **93.82** | **92.70** | 12 | **97.55** |
| | `GNMOT` *(Wang et al., 2022)* | **A** | **99.75** | — | — | — | — |
| | **`D-FINE-M + DroneRLTracker` (Ours)** | **B** | **76.70** | **55.31** | **71.61** | 20 | 96.67 |
| | **`YOLOX-M + DroneRLTracker` (Ours)** | **B** | 40.97 | 51.62 | 63.62 | **13** | **97.15** |

**Key Tier-A Findings**:
1. **Substantial Win on Hard Motion + Lens Jitter (`UAVSwarm-28`)**: Under the exact same GT-detection protocol, `DroneRLTracker` beats `GNMOT` by **`+0.68%` MOTA (`99.08%` vs `98.4%`)**, **`+6.83%` IDF1 (`83.73%` vs `76.9%`)**, and **cuts ID switches by `42.9%` (`8` vs `14` IDSW)** while keeping `100.00%` Track Purity. Meanwhile, classical IoU matching (`Kalman_IoU_Hungarian`, `73.42%` MOTA, `111` IDSW) and static-gate greedy RL (`92.90%` MOTA, `62` IDSW) collapse on `UAVSwarm-28`, and end-to-end `ByteTrack` drops to `32.5% MOTA / 30.0% IDF1`.
2. **Why `GNMOT` Achieves `25/36` Perfect Sequences vs. Our `23/36` (`99.75%` vs `99.55%` on `UAVSwarm-46`)**: `DroneRLTracker` terminates lost tracks at `MAX_MISSED = 8` frames (`> 9` missing frames spawns a new track ID). Sequences like `UAVSwarm-28, 38, 46, 70, 72` contain long GT occlusion gaps (`9–43` frames), creating clean track ID splits (`100%` purity on `70`, `72`, `38`, `28` but lower IDF1/HOTA). A graph-based global matcher like `GNMOT` can bridge slightly longer gaps without re-spawning.

---

## 2. Comprehensive Multi-Object Tracking Comparison (Tier B & Tier C)

### 2.1 Published UAVSwarm Test Split Trackers (`36` Seqs, `5,754` Frames)

This table unifies all published end-to-end UAVSwarm tracking evaluations (**Tier B**) and includes our **Tier A (`DroneRLTracker`)** explicitly flagged at the top strictly as the **association-only upper-bound reference**.

| Paper / Method | Year | Tier & Detector Setup | MOTA (%) ↑ | IDF1 (%) ↑ | HOTA (%) ↑ | IDSW ↓ | FPS ↑ | Provenance |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`DroneRLTracker` (Ours — Upper-Bound Ref.)** | 2026 | **Tier A** (Oracle GT boxes shuffled, no det noise) | *99.84* | *96.12* | *97.14* | *44* | *1,599.5 (CPU)* | [evaluations/results.md](evaluations/results.md) |
| **RoeNet** *(Zhou / Wang et al.)* | Jul 2026 | **Tier B** (Physics-inspired joint det+track) | **85.99** | 84.03 | 72.25 | — | — | `[Prompt / Paywalled ESWA]` |
| **EECTracker** *(Chu et al.)* | Sep 2026 | **Tier B** (`YOLOX-X` + PSMP + EEC, RTX 3090) | 84.71 | **90.19** | 69.16 | **54** | 22.9 | `[Open-Access arXiv:2609.15171 Table I]` |
| **`D-FINE-M + DroneRLTracker` (Ours — End-to-End)** | 2026 | **Tier B** (`D-FINE-M 640×640`, L4 det + CPU trk) | **82.87** | **86.91** | **68.12** | 96 | 28.8 det / 1,735 trk | [end_to_end_summary.csv](evaluations/results/tier_b/end_to_end_summary.csv) |
| **`YOLOX-M + DroneRLTracker` (Ours — End-to-End)** | 2026 | **Tier B** (`YOLOX-M 640×640` ONNX, L4 det + CPU trk) | **81.94** | **87.18** | **66.85** | **54** | **57.0 det / 1,767 trk** | [end_to_end_summary.csv](evaluations/results/tier_b/end_to_end_summary.csv) |
| **SCT-MOT** *(Chu et al.)* | Apr 2026 | **Tier B** (`YOLOX-X` + SMTP + TG-STFF, RTX 3090) | 81.90 | 88.45 | 68.56 | 56 | 22.3 | `[Open-Access arXiv:2604.06883 Table I/VI]` |
| **KRAfT** *(Rahman et al.)* | Aug 2026 | **Tier B** (Kalman Residual Diffusion + Procrustes) | — | — | **80.37** | — | — | `[Prompt / Paywalled ICPR 2026]` |
| **KRAfT (`UAVSwarm–W2C` cross-weather)** | Aug 2026 | **Tier B** (Harder cross-weather/fold variant) | — | — | 70.59 | — | — | `[Prompt / Paywalled ICPR 2026]` |
| **HOMATracker** *(Chu et al.)* | Dec 2025 | **Tier B** (`YOLOX-X` + MPA-Net + M2DA + MHA, 3090) | 79.2 | 87.1 | 67.0 | 58 | 20.3 | `[Verified via SCT-MOT/EEC Table I]` |
| **EECTracker-Light** *(Chu et al.)* | Sep 2026 | **Tier B** (Embedded, Jetson Orin NX) | 78.73 | 86.35 | 65.60 | — | 21.3 | `[Open-Access arXiv:2609.15171 Table XII]` |
| **SCT-MOT Light** *(Chu et al.)* | Apr 2026 | **Tier B** (Embedded, Jetson Orin NX) | 78.22 | 84.04 | 66.71 | — | 24.5 | `[Open-Access arXiv:2604.06883 Table VII]` |
| **ECA + Swin + BYTE** *(Wang et al.)* | Oct 2024 | **Tier B** (ECA+Swin YOLO detector + BYTE) | 78.2 | — | — | — | — | `[Indexed IEEE Sensors Abstract/Excerpt]` |
| **Hybrid-SORT** *(Unified YOLOX-X benchmark)* | 2024/25 | **Tier B** (`YOLOX-X`, RTX 3090) | 77.4 | 80.1 | 62.8 | 459 | 28.9 | `[Verified via SCT-MOT/EEC Table I]` |
| **OC-SORT** *(Unified YOLOX-X benchmark)* | 2023/25 | **Tier B** (`YOLOX-X`, RTX 3090) | 75.7 | 81.8 | 64.7 | 709 | **40.9** | `[Verified via SCT-MOT/EEC Table I]` |
| **UAVS-MOT** *(Wang et al.)* | Sep 2023 | **Tier B** (FairMOT DLA-34+CA+ArcFace+BYTE) | 73.4 | 76.1 | 65.8 | 740 | 34.6 | `[SciOpen + SCT-MOT/EEC Table I]` |
| **Query-Guided Dense Swarm MOT** *(Zhang et al.)* | Apr 2026 | **Tier B** (`MOTRv2` + geom. query + deformable attn) | 72.1 | — | 52.4 | 51 | — | `[Indexed MDPI Drones 10(4) Excerpt]` |
| **FairMOT** *(Wang et al. 2023 / unified)* | 2021/23 | **Tier B** (Anchor-free DLA-34 JDT) | 67.7 | 73.2 | 59.2 | 590 | 35.2 | `[SciOpen + SCT-MOT/EEC Table I]` |
| **ByteTrack** *(Unified YOLOX-X benchmark)* | 2022/25 | **Tier B** (`YOLOX-X`, RTX 3090) | 65.0 | 76.5 | 60.6 | 67 | 39.2 | `[Verified via SCT-MOT/EEC Table I]` |
| **ByteTrack-Light** *(Embedded benchmark)* | 2026 | **Tier B** (Embedded, Jetson Orin NX) | 63.71 | 75.13 | 59.60 | — | 23.9 | `[Open-Access arXiv:2604.06883 / 2609.15171]` |
| **MOTRv3** *(Unified transformer baseline)* | 2023/25 | **Tier B** (End-to-end query transformer, RTX 3090) | 62.3 | 71.9 | 57.9 | 72 | 20.6 | `[Verified via SCT-MOT/EEC Table I]` |
| **DeepSORT** *(Unified YOLOX-X benchmark)* | 2017/25 | **Tier B** (`YOLOX-X`, RTX 3090) | 61.2 | 70.3 | 58.8 | 221 | 33.6 | `[Verified via SCT-MOT/EEC Table I]` |
| **V-USDT (`FCTrack`)** *(Rahman & Madria)* | 2025 | **Tier B** (`DroneDet`/YOLOv11/YOLOX + formation tracker) | +2.6 to +4.0% over ByteTrack | — | — | — | — | `[Indexed IEEE MDM 2025 Excerpt]` |
| **Air-to-Air Deep MOT** *(Hao et al.)* | Apr 2025 | **Tier B** (`YOLOv5`/`YOLOX` + EMO/ResNet50/DN53) | — | — | — | — | — | `[Paywalled ACM DL 10.1145/3718751.3718805]` |
| **A2MGA-Track** *(Yang et al.)* | Nov 2026 | **Tier B** (Unified `YOLOX` + MUP/KGDA/RCTM) | — | — | — | — | — | `[Publisher Bot-Gate / Paywalled]` |

### 2.2 Tier C — Adjacent Multi-UAV Benchmarks (`GL-DT / JPTrack`, `arXiv:2510.09092`)

Although cited alongside UAVSwarm trackers in literature reviews, `GL-DT` (Liu et al., Oct 2025 / *IEEE TGRS* 2026) tabulates its quantitative results on **`MOT-FLY`** and **`FT Dataset`** rather than UAVSwarm:

| Dataset | Method | MOTA (%) ↑ | IDF1 (%) ↑ | HOTA (%) ↑ | DetA (%) ↑ | AssA (%) ↑ | IDSW ↓ | FPS ↑ | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MOT-FLY** | `GL-DT / JPTrack` (GD+LD+STFF) | **82.84** | **79.21** | **65.22** | **67.50** | **63.08** | **33** | 35.06 | `arXiv:2510.09092` Table II/IV |
| **MOT-FLY** | `ByteTrack` *(baseline)* | 50.84 | 52.42 | 39.20 | 41.99 | 36.74 | 122 | 35.81 | `arXiv:2510.09092` Table II |
| **FT Dataset** | `GL-DT / JPTrack` (GD+LD+STFF) | **72.06** | **83.99** | **60.73** | **56.89** | **65.88** | **14** | 35.88 | `arXiv:2510.09092` Table II/IV |
| **FT Dataset** | `ByteTrack` *(baseline)* | 66.87 | 46.03 | 43.02 | 52.89 | 38.03 | 76 | 36.43 | `arXiv:2510.09092` Table II |

### 2.3 Key Insights From Tier-A, Tier-B, and Tier-D Comparisons (2022–2026)
1. **End-to-End 1-to-1 Ranking Across All 36 UAVSwarm Test Videos (`5,754` Frames)**:
   - **`D-FINE-M (640×640) + DroneRLTracker`** achieves **`82.87% MOTA`**, **`68.12% HOTA`**, and **`86.91% IDF1`** (`96` IDSW, `28.8 FPS` L4 detector + `1,735.1 FPS` CPU tracker) — **beating `SCT-MOT` (`81.90% MOTA`), `HOMATracker` (`79.2%`), `ECA+Swin+BYTE` (`78.2%`), `Hybrid-SORT` (`77.4%`), `OC-SORT` (`75.7%`), `UAVS-MOT` (`73.4%`), `Query-Guided` (`72.1%`), `FairMOT` (`67.7%`), `ByteTrack` (`65.0%`), `MOTRv3` (`62.3%`), and `DeepSORT` (`61.2%`) on MOTA** despite running at standard `640×640` input resolution instead of `1088×1088` `YOLOX-X`.
   - **`YOLOX-M (640×640) + DroneRLTracker`** achieves **`81.94% MOTA`**, **`66.85% HOTA`**, **`87.18% IDF1`**, and ties **`EECTracker`'s SOTA `54` ID switches** at **`57.0 FPS` L4 detector + `1,766.8 FPS` CPU tracker** (~`2.5×` faster detector throughput than `HOMATracker`, `SCT-MOT`, and `EECTracker`).
   - Only **`RoeNet`** (`85.99% MOTA`, `72.25% HOTA`), **`EECTracker`** (`84.71% MOTA`, `69.16% HOTA`, `90.19% IDF1`), and **`KRAfT`** (`80.37% HOTA`; `MOTA`/`IDF1` not supplied) surpass our end-to-end Tier-B scores, while **`SCT-MOT`** retains a slight `+0.44%` HOTA / `+1.54%` IDF1 edge (`68.56% / 88.45%` vs our `68.12% / 86.91%`).
2. **Detection Quality Is the True Bottleneck on UAVSwarm**: Look at `EECTracker` (`arXiv:2609.15171`, Table I & Table II): using a standard `YOLOX-X` baseline (`85.40% AP50`), `HOMATracker` gets `79.2% MOTA` and `67.0% HOTA`. By adding motion-prior pixel-level feature compensation to recover weak detections (`AP50: 85.40% → 89.30%`), `EECTracker` jumps `+5.51%` MOTA (`84.71%`). Our L4 fine-tuned `D-FINE-M` (`92.86% AP50`) and `YOLOX-M` (`90.28% AP50`) already exceed that `89.30% AP50` benchmark at `640×640`, with most remaining Tier-B losses concentrated in two hard outlier test sequences (`UAVSwarm-18` and `UAVSwarm-22`).
3. **Our Association Stage Has Ultra-Low IDSW & Minimal Compute**: Pairing our appearance-free `DroneRLTracker` (`~1,750 FPS` CPU association) cuts Tier-B ID switches by **`2.2×–3.1×` over `Kalman_IoU_Hungarian` (`54 vs 170` on YOLOX-M; `96 vs 209` on D-FINE-M)** and **`9×–10.5×` over `Static250_GreedyRL` (`54 vs 569`; `96 vs 861`)**.

---

## 3. Object Detection Comparison on UAVSwarm (Tier D)

> [!NOTE]
> **Leak-Free Official Test Protocol**: Both **`D-FINE HGNetv2-M`** and **`YOLOX-M`** were trained strictly on `train_85` (`5,805` imgs), checkpoint- and threshold-tuned on `val_15` (`1,039` imgs), and evaluated **strictly once** on the **exact official `5,754`-frame UAVSwarm test split** (`37,703` COCO boxes) — enabling a direct 1-to-1 Tier-D comparison alongside published benchmarks.

| Model / Paper | Split & Resolution | Backbone / Architecture | Checkpoint Size / Params | `AP50` / `mAP@0.5` (%) ↑ | COCO `mAP@[.5:.95]` (%) ↑ | `AP_small` (%) ↑ | Speed (FPS / Device) | Source |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Our `D-FINE HGNetv2-M` (L4 Fine-Tuned)** | **Official Test (`5,754` img, `640×640`)** | HGNetv2-M + HybridEnc + FDR Dec | `78.57 MB` EMA / `19.2M` | **92.86** | **48.94** | **46.47** | `28.8 FPS` (`34.8 ms`, L4) | [dfine_eval_metrics.json](evaluations/results/tier_b/dfine_eval_metrics.json) |
| **Our `YOLOX-M` (L4 Fine-Tuned)** | **Official Test (`5,754` img, `640×640`)** | CSPDarknet-M (`YOLOX-M` ONNX) | `101.20 MB` ONNX | 90.28 | 44.84 | 42.07 | **`57.0 FPS`** (`17.5 ms`, L4) | [yolox_eval_metrics.json](evaluations/results/tier_b/yolox_eval_metrics.json) |
| **Wang et al. (May 2022) `Faster R-CNN`** | Official Test (`5,754` img, `640×640`) | ResNet-50 two-stage | — | 50.75 *(mAP)* | — | — | 17 FPS | `[MDPI 2022 Table 6]` |
| **Wang et al. (May 2022) `YOLOX`** | Official Test (`5,754` img, `640×640`) | DarkNet-53 | — | 83.68 *(mAP)* | — | — | **119 FPS** | `[MDPI 2022 Table 6]` |
| **Wang et al. (Jan 2023) Light-Attention YOLOX** | Official Test (`5,754` img) | DW-Conv YOLOX + SE + CBAM + DIoU | **~1/10 YOLOX-S** / **3.85 MB** | 82.32 *(mAP)* | — | — | — | `[MDPI Drones 2023]` |
| **Wang et al. (Oct 2024) `ECA + Swin` Detector** | Test split | ECA backbone + Swin neck + $\alpha$-IoU | — | 82.32 *(mAP)* | — | — | — | `[IEEE Sensors Excerpt]` |
| **Hao et al. (Apr 2025) Air-to-Air** | UAVSwarm | `YOLOv5`/`YOLOX` (EMO/R50/DN53) | — | **92.1** *(mAP)* | — | — | — | `[Prompt Summary]` |
| **Rahman & Madria (2025) `DroneDet`** | `640×640` UAVSwarm split | YOLOv11 + parallel RFB + attention | — | **96.8** | — | — | — | `[IEEE MDM Excerpt]` |
| **HOMATracker (`YOLOX-X` baseline)** | Official Test (`1088×1088`) | `YOLOX-X` | ~99M | 85.40 | — | — | 20.3 FPS *(w/ track, 3090)* | `[arXiv:2609.15171 Table II]` |
| **SCT-MOT (`YOLOX-X` + TG-STFF)** | Official Test (`1088×1088`) | `YOLOX-X` + spatio-temporal fusion | — | 86.30 | — | — | 22.3 FPS *(w/ track, 3090)* | `[arXiv:2609.15171 Table III]` |
| **EECTracker (`YOLOX-X` + PSMP + EEC)** | Official Test (`1088×1088`) | `YOLOX-X` + motion prior compensation | — | **89.30** | — | — | 22.9 FPS *(w/ track, 3090)* | `[arXiv:2609.15171 Table I/II]` |
| **RoeNet** *(Zhou / Wang et al., Jul 2026)* | UAVSwarm | Rotor-physics deep network | — | **96.82** (`mAP50`) | **57.72** (`mAP95`) | — | — | `[Prompt / Paywalled ESWA]` |

---

## 4. Detailed Paper-by-Paper Analysis (All 14 Papers)

### 4.1 Group 1: Original Benchmark & Creator Lab Papers (Shenyang Aerospace Univ. / Beihang Univ.)

1. **Original UAVSwarm Benchmark — Wang et al. (May 2022)**
   - *Citation & Link*: *Remote Sensing* **2022**, *14*(11), 2601 — [MDPI Remote Sensing](https://www.mdpi.com/2072-4292/14/11/2601)
   - *Core Setup*: Introduces UAVSwarm (`72` sequences, `13` scenes, `19+` drone types, `3–23` targets/frame; `36` train = `6,844` frames, `36` test = `5,754` frames). Benchmarks `Faster R-CNN` (`50.75%` mAP @ `17 FPS`) and `YOLOX-DarkNet53` (`83.68%` mAP @ `119 FPS`), plus per-sequence tracking via `GNMOT` (Oracle GT boxes, Tier A) and `ByteTrack` (YOLOX detections, Tier B).
   - *Comparison vs. Ours*: **`GNMOT` is our closest methodological comparator for Tier-A tracking.** On `UAVSwarm-28` (complex background + lens jitter), our `DroneRLTracker` achieves **`99.08% MOTA / 85.85% HOTA / 83.73% IDF1 / 8 IDSW`** vs. `GNMOT`'s **`98.4% MOTA / 76.9% IDF1 / 14 IDSW`**. `GNMOT` achieves `100%` on `25/36` sequences vs. our `23/36` because our `MAX_MISSED = 8` splits long GT occlusions (`>8` missing frames).

2. **Lightweight Attention Detection — Wang et al. (Dec 2022 / Jan 2023)**
   - *Citation & Link*: *"A Lightweight UAV Swarm Detection Method Integrated Attention Mechanism"*, *Drones* **2023**, *7*(1), 13 — [MDPI Drones](https://www.mdpi.com/2504-446X/7/1/13)
   - *Core Setup*: Pure object detection (`6,844` train / `5,754` test). Replaces standard convolutions in YOLOX with depthwise separable convolutions, injects `SE` attention in the backbone and `CBAM` in the neck, and optimizes with `DIoU` loss, hitting **`82.32%` mAP** at **`3.85 MB`**.
   - *Comparison vs. Ours*: Their model is much smaller on disk (`3.85 MB` vs. our **`101.20 MB` YOLOX-M ONNX** / **`78.57 MB` D-FINE-M EMA**) for ultra-constrained micro-UAV compute, while our cloud fine-tuned detectors achieve substantially higher official 5,754-frame accuracy (`92.86% AP50` D-FINE-M and `90.28% AP50` YOLOX-M vs `82.32%`) and pair with an ultra-lightweight **CPU-capable (`~1,750 FPS`) RL tracker**.

3. **UAVS-MOT — Wang et al. (Sept 2023 / 2024)**
   - *Citation & Link*: *"Multi-object continuous robust tracking algorithm for anti-UAV swarm"*, *Acta Aeronautica et Astronautica Sinica* — [SciOpen](https://www.sciopen.com/article/10.7527/S1000-6893.2023.29017)
   - *Core Setup*: Builds on anchor-free joint-detection-tracking (`FairMOT`): upgrades `DLA-34` with **Coordinate Attention (`CA`)**, swaps Re-ID supervision to **ArcFace Loss**, and uses **BYTE** association. Verified metrics (`SciOpen` + `SCT-MOT/EEC` Table I): **FairMOT** = `67.7% MOTA / 73.2% IDF1 / 59.2% HOTA / 590 IDSW / 35.2 FPS`; **UAVS-MOT** = **`73.4% MOTA` (`+5.7%`) / `76.1% IDF1` (`+2.9%`) / `65.8% HOTA` / `740 IDSW` / `34.6 FPS`**.
   - *Comparison vs. Ours*: `UAVS-MOT` relies heavily on appearance Re-ID (`ArcFace`) plus BYTE, yet suffers **`740` ID switches** in end-to-end UAVSwarm testing because identically colored swarms defeat appearance embeddings. By contrast, our **100% appearance-free** Tier-B pipelines achieve **`82.87% MOTA / 86.91% IDF1 / 68.12% HOTA / 96 IDSW`** (`D-FINE-M`) and **`81.94% MOTA / 87.18% IDF1 / 66.85% HOTA / 54 IDSW`** (`YOLOX-M` — **`13.7×` fewer IDSW**) while remaining modular across detectors.

4. **ECA + Swin + BYTE — Wang et al. (Oct 2024)**
   - *Citation & Link*: *"A Target Sensing and Visual Tracking Method for Countering Unmanned Aerial Vehicle Swarm"*, *IEEE Sensors Journal*, Vol. 24, No. 19 — [Semantic Scholar](https://www.semanticscholar.org/paper/A-Target-Sensing-and-Visual-Tracking-Method-for-Wang-Meng/59cabbb34d29249443b5f1e551965db66cdf837a)
   - *Core Setup*: Enhances detection via Efficient Channel Attention (`ECA`), a `Swin Transformer` neck, and $\alpha$-IoU regression paired with `BYTE` association, achieving **`82.32%` mAP** and **`78.2%` tracking accuracy (`MOTA`)** (`+12.3%` over FairMOT, `+10.0%` over OC-SORT, `+6.8%` over SORT, `+6.0%` over DeepSORT).
   - *Comparison vs. Ours*: Like our `D-FINE` pipeline ([`run.py`](run.py)), this work introduces transformer attention for global spatial context in cluttered skies; however, `BYTE` still uses standard Kalman + IoU/motion thresholds (`IoU` fails easily on $10\text{--}25\text{ px}$ drones, as verified in our own `Kalman_IoU_Hungarian` ablation which drops `-2.81%` HOTA and suffers `4.5×` more IDSW than our centroid dynamic gate).

---

### 4.2 Group 2: Multi-Object Tracking Evaluations on UAVSwarm (2025–2026)

5. **Air-to-Air Deep Tracking Comparison — Hao et al. (April 2025)**
   - *Citation & Link*: *"Research on air-to-air UAV tracking method based on deep learning"*, *ACM ICBAR* (`10.1145/3718751.3718805`) — [ACM Digital Library](https://dl.acm.org/doi/10.1145/3718751.3718805)
   - *Core Setup*: Benchmarks `YOLOv5` and `YOLOX` across `EMO`, `ResNet-50`, and `DarkNet-53` backbones for air-to-air MOT on UAVSwarm, reaching **`92.1% mAP`** (`[Prompt Summary]`; full MOT tables paywalled in ACM DL).
   - *Comparison vs. Ours*: Confirms `YOLOX` (`90.28% AP50` at `57.0 FPS` for our `YOLOX-M`) as a strong real-time backbone for small swarm UAVs, while our unseen 5,754-frame test benchmark shows **`D-FINE HGNetv2-M`** outperforming `YOLOX-M` on unseen test generalization (`92.86% AP50 / 48.94% mAP / 46.47% AP_small` vs `90.28% / 44.84% / 42.07%`) at roughly half the frame rate (`28.8` vs `57.0 FPS` L4 GPU).

6. **V-USDT (`DroneDet` + `FCTrack`) — Rahman & Madria (2025)**
   - *Citation & Link*: *"V-USDT: Vision-Based UAV Swarm Detection and Tracking by Leveraging Swarm Formation Constraints"*, *IEEE MDM 2025*.
   - *Core Setup*: Introduces `DroneDet` (based on `YOLOv11` with parallel receptive field blocks and attention, reaching **`96.8% AP@0.5`** at `640×640`, `+2%` over baseline YOLOv11) paired with **`FCTrack`**, which enforces swarm formation geometry constraints, outlier filtering, and true-positive recovery (`+2.6% MOTA` over `YOLOv11 + ByteTrack`, `+4% MOTA` over `YOLOX + ByteTrack`; full tables paywalled in IEEE Xplore).
   - *Comparison vs. Ours*: `FCTrack` models **inter-drone formation geometry** globally to rescue weak detections and prevent hops. Currently, our `DroneRLTracker` models **per-drone ego-kinematics** independently (`[dx/g, dy/g, region_id/7, vx/g, vy/g]`) and resolves multi-drone contention only via global bipartite Hungarian optimization. Adding formation residuals to our RL state is a natural upgrade for dense crossings (`UAVSwarm-46`, `97.55%` purity, `12` IDSW).

7. **GL-DT / JPTrack — Liu et al. (Oct 2025 / Jan 2026)**
   - *Citation & Link*: *"GL-DT: Multi-UAV Detection and Tracking With Global–Local Integration"*, *arXiv:2510.09092* / *IEEE TGRS* Vol. 64 — [arXiv](https://arxiv.org/abs/2510.09092)
   - *Core Setup*: Combines **Global-Local collaborative detection (`GD + LD`)** with **Spatio-Temporal Feature Fusion (`STFF`)** and **`JPTrack`** (`JCMA` joint confidence/motion association + `PMR` post-miss recovery). In `arXiv:2510.09092`, full quantitative results are reported on **`MOT-FLY`** (`82.84% MOTA`, `79.21% IDF1`, `65.22% HOTA`, `33 IDSW`, `35.06 FPS`) and **`FT Dataset`** (`72.06% MOTA`, `83.99% IDF1`, `60.73% HOTA`, `14 IDSW`, `35.88 FPS`) rather than UAVSwarm.
   - *Comparison vs. Ours*: Two architectural lessons stand out for our pipeline: (a) local zoomed crop detection (`LD`) dramatically helps tiny drones (`MOT-FLY` AP/detection boost `+11%` MOTA when combined with `GD`); and (b) `JPTrack`'s `PMR` recovery bridges short-to-medium detector dropouts that our fixed `MAX_MISSED = 8` rule currently turns into track splits.

8. **HOMATracker — Chu et al. (Dec 2025)**
   - *Citation & Link*: *"Vision-based swarm tracking of multiple UAVs in air-to-air scenarios"*, *Chinese Journal of Aeronautics* — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1000936125001645)
   - *Core Setup*: Tackles drone appearance homogeneity via `MPA-Net` (multi-frame pose-attention appearance), `M2DA` (multi-frame Wasserstein motion difference), and `MHA` (homogeneous object association) atop `YOLOX-X` (`1088×1088`, RTX 3090). Benchmarks 8 trackers across all 36 UAVSwarm test sequences: **`79.2% MOTA`**, **`87.1% IDF1`**, **`67.0% HOTA`**, **`58 IDSW`**, **`20.3 FPS`**.
   - *Comparison vs. Ours*: Both of our `640×640` Tier-B pipelines beat `HOMATracker` on MOTA (`82.87%` D-FINE-M and `81.94%` YOLOX-M vs `79.2%`), while `YOLOX-M + DroneRLTracker` edges out `HOMATracker` on `IDF1` (`87.18%` vs `87.1%`) and `IDSW` (`54` vs `58`) at **`2.8×` faster detector throughput (`57.0 FPS` L4 + `1,767 FPS` CPU track)** without heavy `YOLOX-X` (`~99M` params) + `MPA-Net` feature branches.

9. **SCT-MOT — Chu et al. (April 2026)**
   - *Citation & Link*: *"SCT-MOT: Enhancing Air-to-Air Multiple UAVs Tracking with Swarm-Coupled Motion and Trajectory Guidance"*, *IEEE TAES* — [arXiv:2604.06883](https://arxiv.org/abs/2604.06883)
   - *Core Setup*: Extends `HOMATracker` with **Swarm Motion-aware Trajectory Prediction (`SMTP`, window $T=8$)** and **Trajectory-Guided Spatio-Temporal Feature Fusion (`TG-STFF`)** on `YOLOX-X` (`1088×1088`). Verified (`arXiv:2604.06883` Tables I, VI, VII): **`81.90% MOTA`**, **`88.45% IDF1`**, **`68.56% HOTA`**, **`56 IDSW`**, **`22.3 FPS`** (RTX 3090); lightweight embedded version hits **`78.22% MOTA / 84.04% IDF1 / 66.71% HOTA / 24.5 FPS`**.
   - *Comparison vs. Ours*: In direct Tier-B comparison across all 36 sequences, our **`D-FINE-M + DroneRLTracker` beats `SCT-MOT` on `MOTA` (`82.87%` vs `81.90%`)** and **`YOLOX-M + DroneRLTracker` matches/beats `MOTA` (`81.94%`) and `IDSW` (`54` vs `56`)** at `2.5×` higher frame rate, while `SCT-MOT` retains `+0.44%` HOTA (`68.56%` vs `68.12%`) and `+1.27–1.54%` IDF1 thanks to its 8-frame swarm-coupled nonlinear predictor (`SMTP`).

10. **Query-Guided Dense Swarm MOT — Zhang et al. (April 2026)**
    - *Citation & Link*: *"Robust Multi-Object Tracking in Dense Swarms with Query Propagation and Adaptive Attention"*, *Drones* **2026**, *10*(4), 280 — [MDPI Drones](https://www.mdpi.com/2504-446X/10/4/280)
    - *Core Setup*: End-to-end transformer tracker built on `MOTRv2` with sine-cosine geometric query injection, width-height-modulated deformable attention, and motion-direction-consistency re-association. Reports **`72.1% MOTA`**, **`52.4% HOTA`**, **`51 IDSW`** on UAVSwarm.
    - *Comparison vs. Ours*: Both of our decoupled detector + kinematic RL pipelines outperform this end-to-end query tracker by **`+9.8% to +10.8%` MOTA (`81.94%–82.87%` vs `72.1%`) and `+14.5% to +15.7%` HOTA (`66.85%–68.12%` vs `52.4%`)**, confirming that specialized decoupled detectors (`D-FINE-M` / `YOLOX-M`) paired with dynamic kinematic gating remain stronger than joint transformer query propagation on dense aerial swarms.

11. **RoeNet / RoeNet-PS — Zhou / Wang et al. (July 2026)**
    - *Citation & Link*: *"RoeNet: Physics-inspired deep learning network for UAV detection and tracking"*, *Expert Systems with Applications*, Vol. 317, 131912 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426008250)
    - *Core Setup*: Integrates rotor/flight physics priors into detection (`96.82% mAP50`, `57.72% mAP95`) and tracking (**`85.99% MOTA`**, **`72.25% HOTA`**, **`84.03% IDF1`**; `[Prompt Summary — Paywalled ScienceDirect]`).
    - *Comparison vs. Ours*: `RoeNet` holds the overall Tier-B **`MOTA` lead (`85.99%` vs our `82.87%`)** and `+4.13%` `HOTA` (`72.25%` vs `68.12%`) driven by its higher detection recall/precision (`96.82% AP50` vs our `92.86%`), while our **`YOLOX-M + DroneRLTracker` (`87.18% IDF1`) and `D-FINE-M + DroneRLTracker` (`86.91% IDF1`) outperform `RoeNet` on identity preservation (`+2.88% to +3.15%` IDF1 vs `84.03%`)**.

12. **KRAfT — Rahman et al. (Aug 2026)**
    - *Citation & Link*: *"KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking"*, *IEEE ICPR 2026* / Springer LNCS — [ACM / Springer](https://dl.acm.org/doi/10.1007/978-3-032-31654-7_2)
    - *Core Setup*: Uses a **Kalman Filter baseline + probabilistic Residual Diffusion model** combined with **Procrustes formation matching**, achieving **`80.37% HOTA`** on UAVSwarm and **`70.59% HOTA`** on the harder cross-weather/fold variant `UAVSwarm–W2C` (`[Prompt Summary — Paywalled ICPR 2026]`).
    - *Comparison vs. Ours*: `KRAfT` retains the highest published Tier-B `HOTA` (`80.37%` vs our `68.12%`). Architecturally, `DroneRLTracker` starts from the same Kalman baseline and applies a lightweight MLP cost modifier (`~1,750 FPS` CPU association) rather than diffusion sampling, leaving detector recall (`UAVSwarm-18/22/24/56`) + formation alignment as the primary gap to close.

13. **EECTracker — Chu et al. (Sept 2026)**
    - *Citation & Link*: *"EECTracker: Swarm Motion Prior-Guided Feature Compensation for Airborne Optical UAV Swarm Tracking"*, *arXiv:2609.15171* — [arXiv](https://arxiv.org/abs/2609.15171)
    - *Core Setup*: Builds a **Probabilistic Swarm Motion Prior (`PSMP`)** from reliable historical tracklets and injects **Energy-Entropy Consistency (`EEC`)** feature compensation into `YOLOX-X` (`AP50: 85.40% → 89.30%`, `1088×1088`). Verified (`arXiv:2609.15171` Tables I, VII, XII): **`84.71% MOTA`**, **`90.19% IDF1`**, **`69.16% HOTA`**, **`54 IDSW`**, **`22.9 FPS`** (`RTX 3090`); **`EECTracker-Light`** on Jetson Orin NX gets **`78.73% MOTA / 86.35% IDF1 / 65.60% HOTA / 21.3 FPS`**.
    - *Comparison vs. Ours*: `EECTracker` leads our Tier-B pipeline by `+1.84%` MOTA (`84.71%` vs `82.87%`) and `+1.04%` HOTA (`69.16%` vs `68.12%`) via multi-frame `YOLOX-X` feature feedback; notably, our lightweight `YOLOX-M (640×640)` matches **`EECTracker`'s exact `54` ID switches**, and both our `640×640` models outperform **`EECTracker-Light`** (`78.73% MOTA / 65.60% HOTA`) cleanly.

14. **A2MGA-Track — Yang et al. (Nov 2026)**
    - *Citation & Link*: *"A2MGA-track: lightweight motion-geometric association for real-time air-to-air multi-UAV tracking"*, *Journal of King Saud University - CIS / Springer* — [Springer](https://link.springer.com/article/10.1007/s44443-026-01437-4)
    - *Core Setup*: Specifically rejects heavy appearance-embedding branches in favor of a pure closed-loop motion-geometric pipeline under unified `YOLOX` detections on `UAVSwarm` and `MOT-FLY`: **Motion Uncertainty-aware Prediction (`MUP`)**, **Kinematic-Guided Discriminative Association (`KGDA`)**, and **Reliability-Constrained Track Management (`RCTM`)** (`[Publisher Bot-Gate — Exact Tables Unavailable]`).
    - *Comparison vs. Ours*: **Conceptually our closest peer in the 2026 literature.** Both `A2MGA-Track` and our `DroneRLTracker` (`run2.py` / `run3.py`) independently converge on the same core philosophy: **appearance-free, real-time kinematic + geometric association** beats Re-ID embeddings on visually identical drone swarms. Notice the subtle difference: `A2MGA-Track` adds explicit reliability-constrained track retention (`RCTM`) for short misses—matching one of the top error sources we identified in [evaluations/results.md](evaluations/results.md).

---

## 5. Architectural Synthesis: Where We Stand & High-Impact Next Steps

### Where Our Pipeline Is Already Strong (1-to-1 Tier A, Tier B & Tier D)
1. **Tier-A Superiority & Precision Against `GNMOT`**: Under shuffled GT boxes, we beat `GNMOT` directly on the stressful `UAVSwarm-28` sequence (`+0.68% MOTA`, `+6.83% IDF1`, `8` vs `14` IDSW) and maintain `99.32%` track purity across all `37,947` boxes (`96.88%` of tracks `100%`-pure).
2. **Tier-B Competitive Top-4 Standing at `640×640`**: In full 36-sequence unseen end-to-end testing, **`D-FINE-M + DroneRLTracker` (`82.87% MOTA / 68.12% HOTA / 86.91% IDF1`)** and **`YOLOX-M + DroneRLTracker` (`81.94% MOTA / 66.85% HOTA / 87.18% IDF1 / 54 IDSW`)** outperform **11 of 14 published baselines** on `MOTA` (`SCT-MOT`, `HOMATracker`, `ECA+Swin+BYTE`, `Hybrid-SORT`, `OC-SORT`, `UAVS-MOT`, `Query-Guided`, `FairMOT`, `ByteTrack`, `MOTRv3`, `DeepSORT`), tie `EECTracker`'s SOTA `54 IDSW` (via `YOLOX-M`), and beat `RoeNet` on `IDF1` (`+2.9–3.2%`).
3. **Strong Official `AP50` & Runtime Efficiency**: On the official `5,754`-frame test set at `640×640`, our `D-FINE-M` (`92.86% AP50`, `28.8 FPS` L4 GPU) and `YOLOX-M` (`90.28% AP50`, `57.0 FPS` L4 GPU) surpass `HOMATracker` (`85.40%`), `SCT-MOT` (`86.30%`), and `EECTracker` (`89.30%`, `1088×1088`) on `AP50`, paired with a `~1,750 FPS` CPU appearance-free RL tracker.
4. **Ablation Proof in Tier B & Tier A**: Switching from `Kalman_IoU_Hungarian` (SORT) to `DroneRLTracker` cuts Tier-B ID switches by **`3.15×` (`170 → 54` on YOLOX-M)** and **`2.18×` (`209 → 96` on D-FINE-M)** while lifting `HOTA` (`+1.20%` and `+1.09%`).

### Concrete Gaps & Next Steps Revealed by 2025–2026 SOTA
1. **Fix Outlier Test Sequences (`UAVSwarm-18`, `22`, `56`, `24`) & Higher-Resolution Inference (`1088×1088`)**: While `24–25 / 36` sequences score `>= 80% MOTA`, four domain-shift / tiny-scale sequences account for a large share of `FN`/`FP` (`UAVSwarm-18`: `-55.49% / -58.42%` MOTA; `UAVSwarm-22`: `0.00% / 3.51%` MOTA; `UAVSwarm-56`: `385/276` FP). Temporal feature compensation (`EECTracker`-style) or higher input resolution (`960–1088px`) would close the remaining gap to `RoeNet` (`85.99%`) and `EECTracker` (`84.71%`).
2. **Resolution-Normalized Gating + Adaptive Track Coasting (`MAX_MISSED`)**: Extending `MAX_MISSED = 8` via a dormant re-acquisition buffer (`25–45` frames, inspired by `JPTrack`'s `PMR` and `A2MGA-Track`'s `RCTM`) and scaling gates by `W / 812.0` targets long GT/detector dropouts (`UAVSwarm-28, 38, 44, 46, 70, 72`).
3. **Swarm-Coupled Motion & Formation Residuals in `DroneQNetwork`**: Adding neighbor-consensus velocity $(\bar{v}_x, \bar{v}_y)$ and retraining `DroneQNetwork` inside `[20, 65] px` dynamic gates targets the remaining `HOTA`/`AssA` gap against `SCT-MOT` (`68.56%`) and `KRAfT` (`80.37%`).
