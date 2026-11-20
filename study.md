# Drone Swarm RL Trajectory Tracker (`RM_Shrimp_extension`): Complete First-Principles Study Guide & Engineering Reference

> **Purpose of this document**: A **single, self-contained reference** combining every piece of knowledge from `README.md`, `PLAN.md`, `DEVLOG.md`, `evaluations/results.md`, `comparision.md`, and the codebase — explained from **first principles**. It starts from the physics and mathematics of multi-object tracking, walks through every algorithmic pillar (Detection, Kalman Filtering, Dynamic Gating, Polar Subregions, Deep Q-Learning, and Bipartite Hungarian Optimization), documents the complete Phase 0–13 engineering history, NVIDIA L4 & T4 cloud training logs, debugging breakthroughs, 36-sequence Tier-A/D/B & cost-ablation benchmarks, output schemas, and CLI commands so you only need to read `study.md`.

---

## Table of Contents

1. [The Core Problem: Why Tracking Drone Swarms Is Hard](#1-the-core-problem-why-tracking-drone-swarms-is-hard)
2. [From `Shrimp` to `RM_Shrimp_extension`: Architectural Evolution](#2-from-shrimp-to-rm_shrimp_extension-architectural-evolution)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Pillar 1 — Data Engineering & Leakage-Free Splits](#4-pillar-1--data-engineering--leakage-free-splits)
5. [Pillar 2 — Object Detection: D-FINE-M Transformer vs. YOLOX-M CNN](#5-pillar-2--object-detection-d-fine-m-transformer-vs-yolox-m-cnn)
6. [Pillar 3 — State Estimation: The 2D Constant-Velocity Kalman Filter](#6-pillar-3--state-estimation-the-2d-constant-velocity-kalman-filter)
7. [Pillar 4 — Velocity-Adaptive Dynamic Gating & 8-Subregion Geometry](#7-pillar-4--velocity-adaptive-dynamic-gating--8-subregion-geometry)
8. [Pillar 5 — Reinforcement Learning (DQN) for Data Association](#8-pillar-5--reinforcement-learning-dqn-for-data-association)
9. [Pillar 6 — Global Bipartite Optimization via the Hungarian Algorithm](#9-pillar-6--global-bipartite-optimization-via-the-hungarian-algorithm)
10. [Track Lifecycle Management & Trajectory Export](#10-track-lifecycle-management--trajectory-export)
11. [The Three Evaluation Pipelines (`run.py`, `run2.py`, `run3.py`)](#11-the-three-evaluation-pipelines-runpy-run2py-run3py)
12. [Empirical Benchmarks, Training Curves & Ablation Proof](#12-empirical-benchmarks-training-curves--ablation-proof)
13. [Complete Phase-by-Phase Engineering & Decision Log (Phases 0–13)](#13-complete-phase-by-phase-engineering--decision-log-phases-013)
14. [Critical Bugs, Cloud Hurdles & Engineering Solutions](#14-critical-bugs-cloud-hurdles--engineering-solutions)
15. [Output Data Schemas, GUI Controls & CLI Reference](#15-output-data-schemas-gui-controls--cli-reference)
16. [Repository File Map, Tech Stack & Citations](#16-repository-file-map-tech-stack--citations)

---

## 1. The Core Problem: Why Tracking Drone Swarms Is Hard

### 1.1 Detection vs. Tracking (Two Fundamentally Different Questions)

Suppose you have a video consisting of frames $I_1, I_2, \dots, I_T$.

1. **Object Detection** is a *per-frame spatial* problem. Given a single static image $I_t$, a detector outputs an **unordered set** of $N_t$ bounding boxes:
   $$\mathcal{D}_t = \left\{ \mathbf{d}_j^{(t)} = \big(x_1, y_1, x_2, y_2, s_j\big) \right\}_{j=1}^{N_t}$$
   where $s_j \in [0, 1]$ is the confidence score and the center coordinate is:
   $$cx_j = \frac{x_1 + x_2}{2}, \qquad cy_j = \frac{y_1 + y_2}{2}$$
   Crucially, **a detector has no memory**. It does not know that Box #4 in Frame 10 is the same physical drone as Box #11 in Frame 9. In fact, detectors often output boxes in arbitrary order, occasionally miss a drone for 1–2 frames (false negative), or jitter box corners by a few pixels.

2. **Multi-Object Tracking (MOT)** — specifically the **Tracking-by-Detection** paradigm used here — is a *temporal correspondence* problem. Given existing active tracks $\mathcal{T}_{t-1} = \{\tau_1, \dots, \tau_M\}$ from frame $t-1$ and unordered detections $\mathcal{D}_t$ from frame $t$, the tracker must solve:
   - **Data Association**: Which detection $\mathbf{d}_j^{(t)}$ belongs to which existing track $\tau_i$?
   - **Track Birth**: Which unmatched detections are newly entering drones?
   - **Track Maintenance / Coasting**: What happens when a track receives no detection for $k$ frames (occlusion or detector miss)?
   - **Track Death**: When has a drone permanently left the scene?

When solved correctly across all frames, each track $i$ yields a continuous **spatial trajectory**:
$$\tau_i = \Big\{ \big(t, \, cx_t, \, cy_t, \, v_{x,t}, \, v_{y,t}, \, [x_1, y_1, x_2, y_2]_t\big) \Big\}$$

### 1.2 Why Naive Trackers Fail on Aerial Drone Swarms

Tracking pedestrians or cars is already nontrivial, but **high-density UAV swarms** (like the `UAVSwarm` benchmark) break three common assumptions of standard trackers:

1. **Zero Appearance Distinctiveness (Re-ID Failure)**:
   - Pedestrian trackers (DeepSORT, BoT-SORT) rely heavily on appearance embeddings (shirt color, jacket texture).
   - In UAV swarms, all objects belong to **one class** (`drone`), often spanning fewer than $32 \times 32$ pixels against complex sky, urban, or forest backgrounds. Two drones 70 pixels apart look nearly identical pixel-for-pixel. Therefore, **motion and spatial geometry** must carry the disambiguation burden.

2. **The Static Gating Dilemma**:
   - In `UAVSwarm` sequences ($812 \times 428$ pixels at $25\text{ FPS}$), formation spacing between neighboring drones is typically **$60\text{--}90\text{ px}$**, while a hovering or cruising drone moves only **$0.5\text{--}5.0\text{ px/frame}$**, and a fast-maneuvering drone can dash **$> 15\text{ px/frame}$**.
   - If you use a **large fixed search gate** (e.g., $R = 250\text{ px}$), $5\text{--}10$ neighboring drones fall inside every track's search radius simultaneously. In experiments on `UAVSwarm-02`, a $250\text{ px}$ static gate with greedy assignment produced **1,059 erratic jumps** ($>25\text{ px}$) as tracks hopped onto neighbor drones.
   - If you use a **tiny fixed search gate** (e.g., $R = 25\text{ px}$), hovering drones stay isolated, *but* any drone executing a sudden acceleration or suffering a 1–2 frame detector dropout escapes the gate and causes an irreversible track break.

3. **Greedy Assignment Collisions**:
   - Even with good individual scores, if Track A picks its favorite detection first (`argmax`) without coordinating with Track B, Track A can steal Track B's detection when two drones pass near one another, triggering cascading ID switches across the swarm.

`RM_Shrimp_extension` solves all three issues systematically.

---

## 2. From `Shrimp` to `RM_Shrimp_extension`: Architectural Evolution

This project extends the **Shrimp RL Tracker** ([SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)), originally built for aquaculture shrimp detection and virtual line counting (`VectorHysteresisCounter`) from a fixed overhead camera.

### 2.1 Original `Shrimp` Repository Baseline (Reference Architecture)

From the source analysis in `DEVLOG.md`, the base `Shrimp` repository consisted of:
- `shrimp_env.py` (`ShrimpTrackingEnv`): OpenAI Gym-style data association environment fed by CSV (`[frame, track_id, cx, cy, x1, y1, x2, y2, score]`), fixed `gate_radius=150px`, 8 subregions, 5D state `(dx, dy, region_id, vx, vy)` (with velocity from the previous two frames in v2), and 4 random clutter fakes.
- `train_agent.py` (`DQNAgent + QNetwork`): `Linear(5→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1)` MLP, 10k replay buffer, `batch_size=64`, `gamma=0.95`, `epsilon_decay=0.995` (`min=0.05`), `max_norm=1.0` grad clip, normalized by `150.0`.
- `run.py` (`RLTracker`): ONNX detector (`last.onnx`) + `SimpleKF` 2D constant-velocity Kalman filter + per-track gate search + `QNetwork` scoring + `VectorHysteresisCounter` (line-crossing counter) with a 5-frame miss deletion threshold.
- `clean_data.py`: Filtered training trajectories (`min_length=16`, `max_gap=0`, `max_jump=200px`) into `rl_clean_training_data.csv`.
- `annotator.py`, `shrimp_tracker_policy.pth`, `1.gt_img/` (and an earlier OC-SORT with Adaptive Kalman Filter variant `ocsort_adaptive_kalman.py` that adapted process noise $Q$ by innovation magnitude).

### 2.2 Nine Architectural Upgrades & Exact Parameter Mapping

At a mathematical level, both problems ask the same core question: *"Given a predicted target location and $K$ candidate detections inside a local search gate, which candidate is the true continuation of the target?"* However, transferring from slow shrimp in a tank to dense aerial drone swarms required **nine fundamental architectural upgrades** and systematic hyperparameter recalibration:

| Dimension / Parameter | Original `Shrimp` System | `RM_Shrimp_extension` (Drone Swarm) | Why the Change Was Necessary |
| :--- | :--- | :--- | :--- |
| **Operational Goal** | Line-crossing counting (`VectorHysteresisCounter`) | **Continuous 2D Trajectory Tracing** (`.mp4`, `.csv`, `.json`) | Drone swarms require full spatial flight paths per ID, not up/down line counts. |
| **Dataset Format** | Custom shrimp CSV | **UAVSwarm MOT16** $\to$ COCO JSON + cleaned MOT CSV | Needed sequence-level splits for detectors and clean tracks for RL training. |
| **Object Detector** | Single `last.onnx` shrimp model | **Dual Fine-Tuned Detectors**: D-FINE (`HGNetv2-M`, `640×640`) & YOLOX-M (`640×640 ONNX`) | Tiny aerial drones ($<32\text{ px}$) need high small-object AP (`AP_small`) & fast inference. |
| **Benchmark Isolation** | Single inference runner (`run.py`) | **Three Comparative Pipelines**: `run.py`, `run2.py` (GT oracle), `run3.py` (YOLOX) | Separately proves detector quality vs. pure RL tracker upper bound. |
| **Training `gate_radius`** | $150\text{ px}$ | **$250\text{ px}$** | Drones move faster with larger pixel displacement from aerial viewpoints. |
| **Inference Gating** | Fixed $R = 150\text{ px}$ | **Velocity-Adaptive Dynamic Gate** $R_{\text{dynamic}} \in [20, 65]\text{ px}$ | Seals out neighbor drones ($60\text{--}90\text{ px}$ away) during hover while expanding during dashes. |
| **Association Solver** | Greedy per-track `argmax Q` | **Global Hungarian Bipartite Matching** weighted by $Q$-score | Prevents greedy track stealing and crisscross ID swaps in dense formations. |
| **RL Clutter (`num_clutter`)** | 4 random uniform fake points | **4 random fakes + up to 12 real neighboring swarm drones** | Forces the RL agent to reject *kinematically plausible real neighbor drones* (hard negatives). |
| **RL Reward Shaping** | Binary ($+1.0$ / $-1.0$, later $+0.5$ near-miss) | **Three-tier** ($+1.0$ exact, $+0.5$ near-miss $<0.2R = 50\text{ px}$, $-1.0$ wrong) | Smooths gradient signal when neighbor drones lie very close to the true target. |
| **Q-Network Hidden Layers** | `5 → 64 → 32 → 1` | **`5 → 128 → 64 → 1`** | Harder multi-drone disambiguation requires wider representation capacity. |
| **Replay Buffer & Epochs** | $10{,}000$ buffer, $300$ epochs | **$50{,}000$ buffer, $600$ epochs** | UAVSwarm sequences are longer and geometrically more complex. |
| **Cleaning: Min Track Length** | $16$ frames | **$20$ frames** | Ensures sufficient multi-step velocity context for fast aerial targets. |
| **Cleaning: Max Frame Gap** | $0$ frames | **$2$ frames** | Allows up to 1-frame occlusion without breaking valid training tracks. |
| **Cleaning: Max Single Jump** | $200\text{ px}$ | **$400\text{ px}$** | Drones can legitimately displace farther between frames than tank shrimp. |
| **Inference Miss Threshold** | $5$ frames | **$8$ frames (`MAX_MISSED = 8`)** | Drones can experience brief multi-frame occlusions or faint-target drops. |

> **Important engineering detail from `DEVLOG.md` & `train_drone_agent.py`**: Although `DroneTrackingEnv` trains with a wide `gate_radius=250.0` so the agent learns to distinguish true targets from both near and far swarm clutter, all spatial features are **normalized by the gate radius**. This scale invariance allows the exact same trained `DroneQNetwork` to run cleanly at inference time under a shrinking/expanding `R_dynamic ∈ [20, 65] px`!

---

## 3. End-to-End System Architecture

During live tracking on any frame $t$, every pipeline executes the same **5-stage closed loop**:

```
Frame I_t (H x W x 3 BGR)
   │
   ▼
[Stage 1: Candidate Extraction]  ──► Pipeline 1 (run.py):  D-FINE HGNetv2-M Transformer (conf=0.55)
                                 ──► Pipeline 2 (run2.py): Shuffled GT Boxes (No IDs)
                                 ──► Pipeline 3 (run3.py): YOLOX-M (640x640 ONNX Runtime, conf=0.50 / det.txt)
   │  Outputs unordered detections: [{'cx', 'cy', 'box', 'score'}, ...]
   ▼
[Stage 2: Kinematic Prediction & Dynamic Gating] (per active track i)
   │  • SimpleKF.predict() ──► predicted centroid (pred_cx, pred_cy)
   │  • R_dynamic = clip(20.0 + 2.0 * ||v|| + 4.0 * missed, 20.0, 65.0)
   ▼
[Stage 3: Candidate Filtering & Feature Extraction]
   │  • Keep detections with dist <= R_dynamic
   │  • Compute 8-subregion bucket (2 rings x 4 quadrants)
   │  • Build normalized 5D state: [dx/R, dy/R, region_id/7, vx/R, vy/R]
   ▼
[Stage 4: Deep Q-Network Scoring + Global Hungarian Matching]
   │  • DroneQNetwork(5 → 128 → 64 → 1) outputs Q(s_{i,j}) ∈ [16, 20]
   │  • Normalized Q: q_norm = tanh((Q - 17.5) / 2.5) ∈ (-1, 1)
   │  • Cost matrix: C_{i,j} = dist_{i,j} * (1.5 - q_norm) > 0
   │  • scipy.optimize.linear_sum_assignment(C) finds global optimal 1-to-1 pairs
   ▼
[Stage 5: Track Update, Lifecycle & Trajectory Rendering]
      • Matched: EMA velocity smoothing + SimpleKF.update(cx, cy) + log trajectory
      • Unmatched tracks: missed += 1 (delete if missed > 8)
      • Unmatched detections: spawn new track (displayed once hits >= 2)
```

---

## 4. Pillar 1 — Data Engineering & Leakage-Free Splits

The foundation of this project is the **UAVSwarm** dataset (12,598+ images across 13 aerial scenes — sky, urban, forest, rural, indoor — with 19+ UAV types and 3 to 23 simultaneous drones per sequence).

### 4.1 Raw MOT16 Annotation Format

In `UAVSwarm`, ground truth (`gt/gt.txt`) is stored in **MOT16** comma-separated text format, one object instance per line:
```
frame_id, track_id, bb_left, bb_top, width, height, conf, class, visibility
```
- Lines with `conf == 0` denote distractors/ignored regions and are filtered out.
- Bounding boxes use top-left origin `(bb_left, bb_top, width, height)`.

### 4.2 Bridge 1: MOT16 $\to$ COCO JSON (`dataset/mot2coco.py`)

Object detectors (D-FINE and YOLOX) train on individual static frames using **COCO JSON** format (`images`, `annotations`, `categories`). `dataset/mot2coco.py` performs this conversion with two crucial safeguards:

1. **Sequence-Level Splitting (No Temporal Data Leakage)**:
   - Adjacent video frames $I_t$ and $I_{t+1}$ are nearly identical. If you randomly split frames 80/10/10, the validation set contains frames 1/30th of a second away from training frames, causing fake inflated accuracy.
   - Instead, `mot2coco.py` shuffles and splits **at the sequence directory level** (`train2017`, `val2017`, `test2017`) so validation tests generalization to completely unseen flights and backgrounds.
2. **0-Indexed Single-Class Sanitization (Critical Discovery for D-FINE)**:
   - As documented in `DEVLOG.md`, raw UAVSwarm COCO files used `category_id: 1` (inherited from standard COCO `'person'/'pedestrian'`).
   - In `dfine_core`, `dfine_hgnetv2_m_uavswarm.yml` sets `num_classes: 1` and `remap_mscoco_category: False`. PyTorch's cross-entropy loss expects class targets in $[0, C-1] = \{0\}$. Leaving `category_id: 1` causes an out-of-bounds target crash during training.
   - Therefore, training JSONs (`/content/uavswarm_coco/{train.json, val.json, full_train.json}`) were sanitized to `category_id: 0` with class name `'drone'`.
3. **Strict Zero-Leakage Detection Protocol (`train_85` / `val_15` / `test_official`)**:
   - For final NVIDIA L4 (`24 GB`) cloud retraining of both **D-FINE HGNetv2-M** and **YOLOX-M**, the 36 official training sequences were partitioned chronologically per sequence without touching the official test set:
     - **`train_85.json`** (`5,805` images, `42,406` drone boxes): First `85%` of frames inside each of the `36` training sequences.
     - **`val_15.json`** (`1,039` images, `7,255` drone boxes): Trailing `15%` of frames inside each training sequence — used **exclusively** for checkpoint selection (`best_stg2.pth` epoch `14` for `D-FINE-M`; epoch `20` for `YOLOX-M`) and F1-optimal confidence threshold tuning (`0.55` for `D-FINE-M`; `0.50` for `YOLOX-M`).
     - **`test_official.json` / `evaluations/data/uavswarm_test_gt/`** (`5,754` frames across all `36` unseen official test sequences `UAVSwarm-02..72`, `37,703` COCO annotations / `37,947` MOT GT boxes): Held out cleanly and evaluated **strictly once** at those locked `val_15` thresholds.

### 4.3 Bridge 2: MOT16 $\to$ Clean Trajectory CSV (`shrimp_drone/mot_to_csv.py`)

The RL tracker does **not** train on raw pixels — it trains on pure motion trajectories. `shrimp_drone/mot_to_csv.py` extracts `(frame, track_id, cx, cy, x1, y1, x2, y2, score)` across all sequences and filters out broken trajectories before RL training:

| Filter Rule | Threshold | Why It Matters from First Principles |
| :--- | :---: | :--- |
| **`min_length`** | $\ge 20$ frames | Ensures tracks are long enough to provide multi-step velocity context. |
| **`max_gap`** | $\le 2$ frames | Allows at most 1-frame temporary occlusion; rejects long dropouts that corrupt finite-difference velocity $(v_x, v_y)$. |
| **`max_jump`** | $\le 400.0\text{ px}$ | Rejects ground-truth ID glitches or teleportations between frames (up from `200px` in `Shrimp`). |

Running this across the 36 UAVSwarm training sequences converted **355 raw tracks / 49,661 detections** into **323 pristine continuous trajectories / 47,011 training points** (`drone_training_data.csv`).

---

## 5. Pillar 2 — Object Detection: D-FINE-M Transformer vs. YOLOX-M CNN

To avoid tying the RL tracker to a single detector paradigm, the project fine-tuned and benchmarked two fundamentally different object detector families on UAVSwarm.

### 5.1 Detector A: D-FINE (`HGNetv2-M` Hybrid Transformer) — Used in `run.py`

**D-FINE** (*Fine-grained Distribution Refinement for Real-Time Object Detection*, `19.19M` parameters, `56.33 GFLOPS`) is a modern DETR-style (DEtection TRansformer) architecture:

1. **Backbone (`HGNetv2-B2`)**: Extracts multi-scale CNN feature maps at 3 pyramid levels (`return_idx: [1, 2, 3]`, channel widths `[384, 768, 1536]`) with `use_lab: True`.
2. **Hybrid Encoder**: Projects features to `hidden_dim: 256` (`depth_mult: 0.67`) and applies self-attention on the coarsest scale plus cross-scale feature fusion.
3. **D-FINE Transformer Decoder (`num_layers: 4`)**: Instead of regressing raw bounding-box coordinates $(x, y, w, h)$ as brittle point estimates, D-FINE predicts **probability distributions over bounding-box edge offsets** and iteratively refines those distributions across 4 decoder layers (Fine-grained Distribution Refinement, FDR).
4. **Deployment & Environment Safeguards (`run.py` & `dfine_core`)**:
   - The hybrid encoder's positional embeddings (`pos_embed`) have a fixed size of **400 tokens**, corresponding to a $20 \times 20$ feature map at stride 32, i.e., $640 \times 640$ input resolution (`INPUT_W = 640`, `INPUT_H = 640`).
   - At runtime, `DFineDetector` converts BGR $\to$ RGB, resizes to `640x640`, runs `cfg.model.eval()` and `cfg.postprocessor.deploy()` using clean EMA weights from `checkpoints/dfine_drone_weights.pth` (`78.57 MB`, MD5 `cba23409f653b79e347db3c9430a338b`, `SCORE_THRES = 0.55` locked on `val_15`), and scales boxes back to `orig_size = (w, h)`.
   - `dfine_core/src/core/_config.py` and `dfine_core/src/solver/det_engine.py` guard `torch.utils.tensorboard.SummaryWriter` with `TYPE_CHECKING` and a lazy property import in `BaseConfig.writer` so inference runs cleanly on NumPy 2.x without triggering incompatible local TensorFlow imports.

### 5.2 Detector B: YOLOX-M (Anchor-Free Decoupled CNN + ONNX) — Used in `run3.py`

**YOLOX-M** (`depth=0.67, width=0.75`, `25.28M` parameters, `73.73 GFLOPS`) takes a purely convolutional anchor-free approach (upgraded from an earlier `8.94M`-parameter `YOLOX-S` prototype for stronger unseen-scene generalization):

1. **Anchor-Free Grid Prediction**:
   - At $640 \times 640$ input resolution, its Feature Pyramid Network outputs feature maps at 3 strides ($s \in \{8, 16, 32\}$), giving spatial grids of $80\times 80$, $40\times 40$, and $20\times 20$:
     $$80^2 + 40^2 + 20^2 = 6400 + 1600 + 400 = 8{,}400\text{ spatial cells}$$
   - Unlike older anchor-based YOLOs, each cell directly predicts one box without predefined anchor aspect ratios — advantageous for tiny variable-aspect drones.
2. **Decoupled Head**: Splits classification and bounding-box regression into separate convolutional branches so localization gradients do not interfere with objectness classification.
3. **ONNX Output Tensor & Post-Processing (`RuntimeYOLOXDetector` in `run3.py`)**:
   - The input image is **letterbox-resized** (aspect-ratio preserving resize + constant `114` gray padding to the ONNX model's auto-detected `(H, W)` input shape, `640×640`) without BGR$\to$RGB swap (matching YOLOX's BGR training convention).
   - The ONNX session (`checkpoints/yolox_drone_finetuned.onnx`, `101.20 MB`, MD5 `c6a9097d1f490a912c8fa62c774589fe`) produces predictions of shape `(8400, 4 + 1 + C)`:
     - Columns `0:4`: `[cx, cy, w, h]` in letterboxed $640 \times 640$ pixel space.
     - Column `4`: `obj_conf` (probability an object exists in this cell).
     - Columns `5:`: `cls_confs` (for single-class `yolox_drone_finetuned.onnx`, shape is `(8400, 6)`; for 80-class COCO fallback `checkpoints/yolox_s.onnx`, shape is `(8400, 85)`).
   - Final confidence per cell is `score = obj_conf * max(cls_confs)`.
   - Boxes above `conf_thresh` (default **`0.50`**, locked on `val_15`) are divided by `scale` to return to original video coordinates, then deduplicated using **Non-Maximum Suppression (`cv2.dnn.NMSBoxes`, `nms_thresh=0.45`)**.

> **Zero-Leakage Unseen Test Insight (`D-FINE-M` vs. `YOLOX-M` on `5,754` Official Test Frames)**: While `YOLOX-M` reaches higher accuracy on within-sequence `val_15` validation frames (`67.84%` AP / `96.82%` AP50 vs `60.65%` / `95.69%`), **`D-FINE HGNetv2-M` generalizes substantially better across unseen flight domains on `test_official`** — achieving **`48.94%` mAP (`+4.10%`)**, **`92.86%` $\text{AP}_{50}$ (`+2.58%`)**, **`46.14%` $\text{AP}_{75}$ (`+8.02%`)**, **`46.47%` $\text{AP}_{\text{small}}$ (`+4.40%`)**, and **`59.94%` $\text{AR}@100$ (`+8.44%`)** in a `22%` smaller weight file (`78.57 MB` vs. `101.20 MB`). Meanwhile, **`YOLOX-M` runs `2.0×` faster on NVIDIA L4 (`57.03 FPS` / `17.53 ms` vs. `28.77 FPS` / `34.76 ms`)** and matches its stability when paired with `DroneRLTracker` (`87.18%` IDF1, `54` IDSW vs. `96` IDSW).

---

## 6. Pillar 3 — State Estimation: The 2D Constant-Velocity Kalman Filter

### 6.1 Why Raw Detections Need Filtering

Suppose Drone #1 is flying right at $4\text{ px/frame}$. Between frame $t-1$ and frame $t$:
- Where should we look for Drone #1? **Not** at its *previous* position $(cx_{t-1}, cy_{t-1})$, because it has moved! We should look at its **predicted position** $(cx_{t-1} + v_x, \, cy_{t-1} + v_y)$.
- What if the detector jittered the box edge by $3\text{ px}$, or missed the drone in frame $t$? We need a principled mathematical filter that fuses noisy measurements with physical motion predictions, weighted by uncertainty.

That is precisely what **`SimpleKF`** (in `run.py`, `run2.py`, `run3.py`) does.

### 6.2 State Space & Linear Kinematics

For each drone track, `SimpleKF` maintains a 4D state vector $\mathbf{x}_t \in \mathbb{R}^4$ and a $4\times 4$ state covariance matrix $\mathbf{P}_t$:
$$\mathbf{x}_t = \begin{bmatrix} cx_t \\ cy_t \\ v_{x,t} \\ v_{y,t} \end{bmatrix}, \qquad \mathbf{P}_0 = 50.0 \, \mathbf{I}_4$$

Assuming approximately constant velocity over a single frame step ($\Delta t = 1\text{ frame}$):
$$cx_t = cx_{t-1} + v_{x,t-1}, \qquad cy_t = cy_{t-1} + v_{y,t-1}, \qquad v_{x,t} = v_{x,t-1}, \qquad v_{y,t} = v_{y,t-1}$$

In matrix form, the **State Transition Matrix** $\mathbf{F}$ and **Measurement Matrix** $\mathbf{H}$ (we observe only centroid position $(cx, cy)$, not velocity directly) are:
$$\mathbf{F} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \qquad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

### 6.3 The Two-Step Kalman Recursion

1. **Predict Step (`kf.predict()`)** — Executed every frame *before* data association:
   $$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F} \mathbf{x}_{t-1}$$
   $$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1} \mathbf{F}^T + \mathbf{Q}$$
   where $\mathbf{Q} = 1.0 \, \mathbf{I}_4$ is the **process noise covariance** (accounting for real drone accelerations/turns that violate constant velocity). Even if a drone is occluded (`missed > 0`), calling `predict()` every frame coasts the predicted position forward along its trajectory while steadily growing $\mathbf{P}_{t|t-1}$ (uncertainty).

2. **Update Step (`kf.update(cx, cy)`)** — Executed only if the track is matched to detection $\mathbf{z}_t = [cx, cy]^T$:
   - **Innovation (Measurement Residual)**: how far the detection landed from the prediction:
     $$\mathbf{y}_t = \mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t|t-1}$$
   - **Innovation Covariance**: total uncertainty (prediction uncertainty + detector noise $\mathbf{R} = 5.0 \, \mathbf{I}_2$):
     $$\mathbf{S}_t = \mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^T + \mathbf{R}$$
   - **Optimal Kalman Gain** $\mathbf{K}_t \in \mathbb{R}^{4 \times 2}$: decides how much to trust the measurement vs. the physics prediction:
     $$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^T \mathbf{S}_t^{-1}$$
   - **Posterior State & Covariance Update**:
     $$\mathbf{x}_t = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \mathbf{y}_t, \qquad \mathbf{P}_t = (\mathbf{I}_4 - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}$$

### 6.4 Smoothed Velocity for Gating & RL Features

Alongside `SimpleKF`, `DroneRLTracker` maintains an Exponential Moving Average (EMA) velocity for stable gate sizing and RL feature inputs:
$$v_x^{\text{raw}} = cx_t - cx_{t-1}, \qquad v_y^{\text{raw}} = cy_t - cy_{t-1}$$
$$v_{x,t} = 0.7 \, v_{x,t-1} + 0.3 \, v_x^{\text{raw}}, \qquad v_{y,t} = 0.7 \, v_{y,t-1} + 0.3 \, v_y^{\text{raw}}$$
Why EMA? Single-frame centroid differences contain detector jitter; a 70/30 EMA damps box jitter so the dynamic gate doesn't flicker wildly frame-to-frame.

---

## 7. Pillar 4 — Velocity-Adaptive Dynamic Gating & 8-Subregion Geometry

### 7.1 Deriving the Dynamic Gate Equation from Swarm Physics

Consider a drone $i$ whose predicted position is $\hat{\mathbf{p}}_i = (\hat{cx}_i, \hat{cy}_i)$, smoothed velocity is $\mathbf{v}_i = (v_x, v_y)$, and consecutive missed frames is $k_{\text{missed}} \ge 0$.

What is the physically valid search radius $R_{\text{dynamic}}$ around $\hat{\mathbf{p}}_i$?
1. **Base Envelope ($R_{\text{base}} = 20.0\text{ px}$)**: Even a completely stationary hovering drone has minor detector jitter and slight wind drift. $20\text{ px}$ encloses the drone's own bounding box without reaching neighbors.
2. **Velocity Scaling ($\alpha \cdot \|\mathbf{v}\|_2$, with $\alpha = 2.0$)**: A faster drone experiences larger unmodeled accelerations (turns, gusts). Scaling the radius linearly with speed $\|\mathbf{v}\|_2 = \sqrt{v_x^2 + v_y^2}$ accommodates rapid maneuvers.
3. **Occlusion / Dropout Expansion ($\beta \cdot k_{\text{missed}}$, with $\beta = 4.0$)**: Every missed frame adds linear drift uncertainty between the constant-velocity Kalman forecast and the true curved path, so the search cone widens by $4\text{ px}$ per missed frame.
4. **Hard Physical Bounds ($R_{\text{min}} = 20.0\text{ px}$, $R_{\text{max}} = 65.0\text{ px}$)**: Crucially, $R_{\text{max}} = 65\text{ px}$ is tuned to remain **below the $60\text{--}90\text{ px}$ typical inter-drone spacing** in formation, physically preventing long-range neighbor capture.

Combining all four terms yields `compute_dynamic_gate()` (and `get_dynamic_gate()` in `drone_env.py`):
$$R_{\text{dynamic}}(\mathbf{v}, k) = \mathrm{clip}\Big( 20.0 + 2.0\sqrt{v_x^2 + v_y^2} + 4.0 \, k_{\text{missed}}, \;\; 20.0, \;\; 65.0 \Big)$$

| Flight Regime | Speed $\|\mathbf{v}\|_2$ | Missed Frames $k$ | $R_{\text{dynamic}}$ | Physical Effect |
| :--- | :---: | :---: | :---: | :--- |
| **Hovering / Stationary** | $\le 0.5\text{ px/f}$ | $0$ | **$21.0\text{ px}$** | Tightly sealed; neighbors $60\text{--}90\text{ px}$ away are physically excluded. |
| **Cruising Formation** | $3.0\text{ px/f}$ | $0$ | **$26.0\text{ px}$** | Follows formation motion cleanly with zero neighbor leakage. |
| **High-Speed Dash** | $15.0\text{ px/f}$ | $0$ | **$50.0\text{ px}$** | Expands smoothly to catch aggressive acceleration. |
| **Temporary Occlusion** | $2.0\text{ px/f}$ | $2$ | **$32.0\text{ px}$** | Widens search window to re-acquire drone immediately upon re-detection. |

### 7.2 Polar 8-Subregion Spatial Discretization (`get_subregion`)

For every candidate detection $j$ inside the gate ($d = \sqrt{\Delta x^2 + \Delta y^2} \le R_{\text{dynamic}}$, where $\Delta x = cx_j - \hat{cx}_i, \; \Delta y = cy_j - \hat{cy}_i$), the tracker maps $(\Delta x, \Delta y)$ into **8 discrete polar subregions** ($2\text{ concentric rings} \times 4\text{ angular quadrants}$):

1. **Radial Ring**:
   $$\text{ring} = \begin{cases} 0 & \text{if } d \le R / 2 \quad (\text{Inner ring: high spatial proximity}) \\ 1 & \text{if } R/2 < d \le R \quad (\text{Outer ring: peripheral candidates}) \end{cases}$$
2. **Angular Quadrant** via $\theta = \mathrm{atan2}(\Delta y, \Delta x) \in (-\pi, \pi]$:
   - $q = 0$ if $-\frac{\pi}{4} \le \theta < \frac{\pi}{4}$ (Right)
   - $q = 1$ if $\frac{\pi}{4} \le \theta < \frac{3\pi}{4}$ ($+\Delta y$ direction; note image coordinates increase downward, though comments preserve the Shrimp label)
   - $q = 2$ if $\theta \ge \frac{3\pi}{4}$ or $\theta < -\frac{3\pi}{4}$ (Left)
   - $q = 3$ otherwise (Opposite vertical direction)
3. **Subregion ID**:
   $$\text{region\_id} = 4 \cdot \text{ring} + q \in \{0, 1, 2, 3, 4, 5, 6, 7\}$$

```
                       q=3 (-3pi/4 .. -pi/4)
                     \       Outer [7]       /
                       \     Inner (3)     /
     q=2 (Left)          +---------------+          q=0 (Right)
   Outer [6] | Inner (2) |  Pred Center  | Inner (0) | Outer [4]
                         +---------------+
                       /     Inner (1)     \
                     /       Outer [5]       \
                       q=1 (+pi/4 .. +3pi/4)

   Inner ring (d <= R/2): region_id = 0, 1, 2, 3
   Outer ring (d >  R/2): region_id = 4, 5, 6, 7
```

**Why include `region_id` alongside continuous $(\Delta x, \Delta y)$?**  
A small 3-layer MLP benefits hugely from explicit polar/radial bucketing: it immediately sees whether a candidate sits inside the inner core (`0..3`) or outer periphery (`4..7`) and whether its angular sector aligns with the track's velocity quadrant!

---

## 8. Pillar 5 — Reinforcement Learning (DQN) for Data Association

### 8.1 Why Frame Data Association as a Reinforcement Learning Problem?

Why not just match detections purely by Euclidean distance $d_{i,j}$?
Consider two drones flying parallel or crossing paths:
- Suppose Drone A is moving right at $(v_x=+8, v_y=0)$, and Drone B happens to be $10\text{ px}$ *behind* Drone A's predicted position ($\Delta x = -10$), while Drone A's true detection jittered $12\text{ px}$ *forward* ($\Delta x = +12$).
- Pure Euclidean distance sees $10 < 12$ and picks **the wrong drone**!
- A learned spatial-temporal policy sees the velocity vector $(v_x, v_y)$ together with $(\Delta x, \Delta y, \text{region\_id})$ and learns that errors **aligned with momentum** are far more likely to be the true continuation than detections behind the track or in neighboring formation lanes.

### 8.2 Variable-Action-Set Formulation (`DroneTrackingEnv`)

In standard Atari DQN, the action set is fixed (e.g., 4 joystick moves), so a network outputs $Q(s, a_1), \dots, Q(s, a_4)$ simultaneously.  
In Multi-Object Tracking, the number of candidates $K_t$ inside a gate **changes every frame** ($K_t$ can be 1, 5, or 17).

**The Shrimp/Drone Trick**: Instead of a fixed-output network, `DroneQNetwork` evaluates **each candidate $j \in \{1, \dots, K_t\}$ independently** using a shared scoring MLP:
$$\mathbf{s}_{i,j} = \left[ \frac{\Delta x}{R}, \;\; \frac{\Delta y}{R}, \;\; \frac{\text{region\_id}}{7.0}, \;\; \frac{v_x}{R}, \;\; \frac{v_y}{R} \right]^T \in \mathbb{R}^5$$
$$Q_\theta(\mathbf{s}_{i,j}) = \text{MLP}_{5 \to 128 \to 64 \to 1}(\mathbf{s}_{i,j}) \in \mathbb{R}$$

Notice two subtle architectural choices documented in `shrimp_drone/train_drone_agent.py`:
1. **No BatchNorm**: Because a frame at inference might have only $K=1$ candidate inside a gate, BatchNorm would fail or behave inconsistently. Normalization is done analytically by dividing by `gate_radius` (and `7.0` for `region_id`).
2. **Scale Invariance**: During training (`gate_radius = 250.0`), dividing $(\Delta x, \Delta y, v_x, v_y)$ by $R$ maps coordinates into roughly $[-1, 1]$. At inference (`run.py`/`run2.py`/`run3.py`), dividing by $R_{\text{dynamic}} \in [20, 65]$ keeps the input distribution dimensionless and consistent!

### 8.3 Hard-Negative Swarm Clutter & Shaped Reward (`drone_env.py` & `train_drone_agent.py`)

In `DroneTrackingEnv.step()`, each training step constructs a realistic crowded decision:
1. **4 Random Uniform Clutter Candidates**: fake points uniformly sampled in polar coordinates within $R = 250\text{ px}$ (`_random_clutter`).
2. **Real Swarm Clutter Hard Negatives (`_swarm_clutter`)**: looks up **all other real drones present in the exact same video frame** within $R = 250\text{ px}$ (`det['track_id'] != self.current_track_id`). This exposes the Q-network to real formation geometry (e.g., neighbors $60\text{--}90\text{ px}$ away moving in parallel)!
3. **1 Ground-Truth Candidate**: the actual next position of the tracked drone (`true_cx, true_cy`), shuffled into a random index `true_index` so order gives zero hints.

When the agent picks candidate index $a \in \{0, \dots, K_t - 1\}$, `compute_reward()` returns:
$$r(a) = \begin{cases} +1.0 & \text{if } a = \text{true\_index} \quad (\text{exact match}) \\ +0.5 & \text{if } \|\mathbf{p}_a - \mathbf{p}_{\text{true}}\|_2 < 0.2 \cdot R \quad (\text{near miss: within } 50\text{ px at } R=250) \\ -1.0 & \text{otherwise} \quad (\text{wrong drone or clutter}) \end{cases}$$

**Why $+0.5$ for near-misses?** In dense swarms, two detections may sit very close together. Punishing a tiny geometric near-miss with $-1.0$ (the exact same penalty as picking background noise $200\text{ px}$ away) produces harsh, noisy gradients. Graded partial credit shapes a smoother Q-value landscape.

### 8.4 Deep Q-Learning Training Loop

`DroneAgent` trains `DroneQNetwork` over 600 episodes (each episode walks one cleaned drone trajectory from `drone_training_data.csv`):
- **Exploration**: $\epsilon$-greedy selection ($\epsilon$ decays exponentially from $1.0$ down to $0.05$ with `epsilon_decay = 0.995` per episode).
- **Experience Replay**: `ReplayBuffer` stores up to $50{,}000$ transitions $(S_t, a_t, r_t, S_{t+1}, \text{done}_t)$.
- **Bellman TD Target** ($\gamma = 0.95$):
  $$y_t = \begin{cases} r_t & \text{if } \text{done}_t \text{ or } S_{t+1} \text{ is None} \\ r_t + \gamma \max_{a' \in S_{t+1}} Q_\theta(S_{t+1}, a') & \text{otherwise} \end{cases}$$
- **Optimization**: `MSELoss(Q_\theta(S_t, a_t), y_t)`, Adam optimizer (`lr = 1e-3`), gradient clipping (`max_norm = 1.0`), batch size $64$. Best weights by episode association accuracy are saved to `checkpoints/drone_tracker_policy_best.pth` (`39.8 KB`).

---

## 9. Pillar 6 — Global Bipartite Optimization via the Hungarian Algorithm

Having a smart Q-network solves *per-track scoring*, **but** if $M$ tracks score $N$ detections independently and take greedy `argmax`, two tracks can still claim the same detection or fight over neighbors.

### 9.1 Formulating Swarm Association as Minimum-Weight Bipartite Matching

Think of the active tracks $\mathcal{T} = \{1, \dots, M\}$ as nodes on the left side of a bipartite graph, and the current frame's detections $\mathcal{D} = \{1, \dots, N\}$ as nodes on the right side. We want a **one-to-one matching** (each track gets at most one detection; each detection goes to at most one track) that minimizes total global assignment cost.

`DroneRLTracker.process_frame()` builds an $M \times N$ cost matrix $\mathbf{C}$ initialized to $10^5$ (forbidden edge), first normalizing the $\gamma = 0.95$ cumulative Q-value into $(-1, 1)$:
$$\tilde{Q}_\theta(\mathbf{s}_{i, j}) = \tanh\!\left(\frac{Q_\theta(\mathbf{s}_{i, j}) - 17.5}{2.5}\right)$$
$$C_{i, j} = \begin{cases} d_{i, j} \cdot \Big( 1.5 - \tilde{Q}_\theta(\mathbf{s}_{i, j}) \Big) & \text{if } d_{i, j} \le R_{\text{dynamic}}^{(i)} \\ 10^5 & \text{otherwise} \end{cases}$$
where $d_{i, j} = \|\mathbf{p}_j - \hat{\mathbf{p}}_i\|_2$ is the Euclidean distance from track $i$'s Kalman prediction to detection $j$.

### 9.2 Understanding the Cost Formula $d_{i,j} \cdot (1.5 - \tilde{Q})$

Look at how normalizing $Q \to \tilde{Q} \in (-1, 1)$ lets spatial physics and RL policy confidence work together multiplicatively:
- Because `DroneAgent` trains with discount factor $\gamma = 0.95$ over multi-step trajectories earning $+1.0$ reward per step, raw `DroneQNetwork` outputs reflect discounted cumulative returns $Q \in [16, 20]$ (approaching $\frac{1}{1-\gamma} = 20.0$).
- Centering and bounding via $\tilde{Q} = \tanh\!\big((Q - 17.5)/2.5\big) \in (-1, 1)$ guarantees $(1.5 - \tilde{Q}) \in (0.5, 2.5)$ is **strictly positive**:
  - When the Q-network is highly confident detection $j$ is the true match ($\tilde{Q} \approx +1.0$):
    $$1.5 - 1.0 = 0.5 \implies C_{i,j} = 0.5 \, d_{i,j} \quad (\text{cost cut in half — strong pull})$$
  - When the Q-network is neutral ($\tilde{Q} \approx 0.0$):
    $$1.5 - 0.0 = 1.5 \implies C_{i,j} = 1.5 \, d_{i,j} \quad (\text{pure distance penalty})$$
  - When the Q-network classifies detection $j$ as clutter or a neighbor drone ($\tilde{Q} \approx -1.0$):
    $$1.5 - (-1.0) = 2.5 \implies C_{i,j} = 2.5 \, d_{i,j} \quad (\text{2.5}\times\text{ penalty — pushes assignment away})$$

### 9.3 Exact Global Solution & Post-Match Sanity Guard

`scipy.optimize.linear_sum_assignment(cost_matrix)` runs the **Hungarian (Kuhn–Munkres) algorithm** in $O(\max(M, N)^3)$ time to find the exact global minimum assignment:
$$\min_{\mathbf{X} \in \{0, 1\}^{M \times N}} \sum_{i=1}^M \sum_{j=1}^N C_{i,j} X_{i,j} \quad \text{subject to} \quad \sum_{j=1}^N X_{i,j} \le 1, \;\; \sum_{i=1}^M X_{i,j} \le 1$$

Finally, every matched pair $(r, c)$ passes two strict filters:
1. $C_{r, c} < 10^4$ (ensures the pair was actually inside the dynamic gate, not a dummy $10^5$ edge forced by matrix rectangularity).
2. **Single-step physical jump check**:
   $$\text{step\_d} = \|\mathbf{p}_c - \mathbf{p}_{\text{last}}^{(r)}\|_2 \le R_{\text{dynamic}}^{(r)} + 5.0\text{ px}$$
   If a Kalman coast drifted slightly, this prevents an unrealistic single-frame jump relative to the last confirmed observation.

*(For a controlled 36-sequence comparison against alternative Hungarian cost formulations `-q`, `-q/(d+eps)`, `d/q`, and pure `dist` in `evaluations/testing_different_stratergies/`, see **Section 12.5**.)*

---

## 10. Track Lifecycle Management & Trajectory Export

Across frames, every drone passes through a deterministic 4-state lifecycle inside `DroneRLTracker.process_frame()`:

1. **Initialization (Frame 1 or Unmatched Detection)**:
   - On the very first frame (`not tids`), or whenever a detection $j \notin \text{matched\_dets}$ is left unmatched by the Hungarian solver, a new track `tid = self.next_id` is spawned with `SimpleKF(cx, cy)`, `hits = 1`, `missed = 0`, `vx = 0.0`, `vy = 0.0`.
2. **Tentative $\to$ Confirmed (`hits >= 2`)**:
   - A brand-new detection could be a 1-frame detector false alarm. Therefore, `process_frame()` only adds tracks with **`hits >= 2`** to the rendered `active` list (meaning the drone had to be detected and successfully associated across at least 2 frames).
3. **Coasting During Occlusion (`missed += 1`)**:
   - If an existing track receives no valid match in frame $t$, its `missed` counter increments. It is **kept alive** in `self.tracks`, its `SimpleKF.predict()` coasts its estimated position forward, and its dynamic gate widens by $+4.0\text{ px}$ per missed frame so it can re-capture the drone when it reappears.
4. **Termination (`missed > MAX_MISSED = 8`)**:
   - If a track goes unobserved for 9 consecutive frames, it is deleted from `self.tracks`. (Note: its already-recorded coordinates remain safely archived in `self.trajectories[tid]`!).

### Output Formats
At the end of the video, any track with at least 2 recorded points (`len(pts) >= 2`) is exported to:
- **Tracked MP4 Video**: Renders the last `TRAIL_LENGTH = 80` centroid points as anti-aliased (`cv2.LINE_AA`) deterministic neon trails, dynamic gate rings (`gate_r`), bounding boxes, ID badges, and a telemetry HUD.
- **CSV (`trajectories_*.csv`)**: Flat table with columns `track_id, frame, cx, cy, vx, vy, x1, y1, x2, y2` (`run2.py` also includes `gt_id`).
- **JSON (`trajectories_*.json`)**: Full dictionary keyed by `track_id` containing per-frame coordinate/velocity dicts.

---

## 11. The Three Evaluation Pipelines (`run.py`, `run2.py`, `run3.py`)

A core scientific strength of `RM_Shrimp_extension` is its **three-pillar experimental design**, separating detector errors from tracker intelligence:

```
                              Input Drone Video Frame
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
   [ Pipeline 1 ]                  [ Pipeline 2 ]                  [ Pipeline 3 ]
       run.py                          run2.py                         run3.py
 (Fine-Tuned D-FINE-M)         (Ground Truth Oracle)          (YOLOX-M Runtime/Bench)
         │                               │                               │
   HGNetv2-M PyTorch           GT Boxes w/ IDs Stripped      YOLOX-M ONNX (conf>=0.50) OR
   (640x640, conf>=0.55)        & Shuffled Every Frame         Official det.txt
         │                               │                               │
         └───────────────────────┬───────┴───────────────────────────────┘
                                 │  Unified Detection Format:
                                 │  [{'cx', 'cy', 'box', 'score'}]
                                 ▼
                  ┌─────────────────────────────┐
                  │      DroneRLTracker         │
                  │  • SimpleKF                 │
                  │  • R_dynamic ∈ [20, 65] px  │
                  │  • 8-Subregion Encoding     │
                  │  • DroneQNetwork (RL)       │
                  │  • Hungarian Matcher        │
                  └──────────────┬──────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
result_with_our_dfine/   result_without_custom_   result_with_yolox/
                          object_detection/
```

### Why Each Pipeline Exists & Full Three-Pillar Architecture Matrix

1. **Pipeline 1 — `run.py` (`result_with_our_dfine/`)**:
   - **Purpose**: End-to-end autonomous deployment using our fine-tuned **D-FINE HGNetv2-M** detector (`checkpoints/dfine_drone_weights.pth`). Works on any arbitrary drone video without annotations (`--video <any.mp4>`). Also supports optional `--benchmark-gt`.
2. **Pipeline 2 — `run2.py` (`result_without_custom_object_detection/`)**:
   - **Purpose**: **Pure Tracker Upper-Bound Benchmark**. Reads ground-truth boxes (`uavswarm_*_gt.txt`), keeps only single-frame boxes (`gt_id` is *never* passed into `DroneRLTracker`'s matching logic; it is only carried as passive metadata on the box for display/comparison), and explicitly runs **`random.shuffle(detections)`** on every frame so list order leaks zero identity information.
   - Proves mathematically that trajectory construction, ID persistence, and zero jumps come from our Kalman + Dynamic Gate + RL + Hungarian tracker itself.
3. **Pipeline 3 — `run3.py` (`result_with_yolox/`)**:
   - **Purpose**: Evaluates the faster **YOLOX-M (`640×640`)** detector in two modes:
     - `--mode runtime`: Live ONNX Runtime inference using our fine-tuned `checkpoints/yolox_drone_finetuned.onnx` (auto-detects input shape; falls back to COCO `checkpoints/yolox_s.onnx` if absent).
     - `--mode benchmark`: Loads the official precomputed `UAVSwarm` benchmark YOLOX detections (`results/uavswarm_*_yolox_det.txt`) for apples-to-apples literature comparison.

| Metric / Property | Pipeline 1 (`run.py`: D-FINE-M + RL) | Pipeline 2 (`run2.py`: GT + RL) | Pipeline 3 (`run3.py`: YOLOX-M + RL) |
| :--- | :---: | :---: | :---: |
| **Detector** | D-FINE HGNetv2-M Transformer | Oracle Bounding Boxes (Shuffled, No IDs) | Fine-Tuned YOLOX-M CNN (`640×640`) |
| **Parameters / Compute** | **$19.19\text{ M}$ (`56.33 GFLOPS`)** | N/A (Ground Truth) | $25.28\text{ M}$ (`73.73 GFLOPS`) |
| **Artifact Size** | **$78.57\text{ MB}$ (EMA `.pth`)** | N/A | $101.20\text{ MB}$ (`.onnx`) |
| **GPU Latency (NVIDIA L4)** | $34.76\text{ ms}$ ($28.77\text{ FPS}$) | $< 1\text{ ms}$ (direct memory) | **$17.53\text{ ms}$ ($57.03\text{ FPS}$)** |
| **Official Test $\text{mAP}@[0.50:0.95]$** | **$48.94\%$** | $100.0\%$ (Upper bound) | $44.84\%$ |
| **Official Test $\text{AP}_{50}$** | **$92.86\%$** | $100.0\%$ | $90.28\%$ |
| **Official Test $\text{AP}_{75}$** | **$46.14\%$** | $100.0\%$ | $38.12\%$ |
| **Official Test $\text{AP}_{\text{small}}$ ($<32\text{ px}$)** | **$46.47\%$** | $100.0\%$ | $42.07\%$ |
| **Locked `val_15` Conf Threshold** | `0.55` (`val_15 F1 = 94.25%`) | N/A | `0.50` (**`val_15 F1 = 96.62%`**) |
| **36-Seq MOT (`MOTA / HOTA / IDF1 / IDSW`)** | **`82.87% / 68.12% / 86.91% / 96`** | `99.84% / 97.14% / 96.12% / 44` | `81.94% / 66.85% / 87.18% / 54` |
| **Primary Output Folder** | `result_with_our_dfine/` | `result_without_custom_object_detection/` | `result_with_yolox/` |
| **Ideal Use Case** | Highest unseen accuracy & global context | Empirical upper-bound tracking evaluation | **$2.0\times$ faster** real-time deployment & low IDSW |

---

## 12. Empirical Benchmarks, Training Curves & Ablation Proof

### 12.1 Detector Comparison (Tier D — `val_15` & Unseen `5,754`-Frame `test_official`, NVIDIA L4)

Trained under strict zero test-set leakage (`train_85`: `5,805` images / `42,406` boxes; `val_15`: `1,039` images / `7,255` boxes for checkpoint & confidence selection) and evaluated once on the **official unseen `UAVSwarm/test` split (`5,754` frames across `36` sequences, `37,703` COCO annotations)** on **NVIDIA L4 (24 GB)** ([evaluations/results.md](evaluations/results.md), [dfine_eval_metrics.json](evaluations/results/tier_b/dfine_eval_metrics.json), [yolox_eval_metrics.json](evaluations/results/tier_b/yolox_eval_metrics.json)):

| Performance Dimension | Split | Fine-Tuned `D-FINE HGNetv2-M` | Fine-Tuned `YOLOX-M (640×640)` | Difference / Winner |
| :--- | :--- | :---: | :---: | :--- |
| **Architecture & Params** | — | Hybrid Transformer (`19.19M` / `56.33 GFLOPS`) | Decoupled CNN (`25.28M` / `73.73 GFLOPS`) | **D-FINE has `24%` fewer params** |
| **Checkpoint / Artifact Size** | — | **`78.57 MB`** EMA `.pth` (`cba23409...`) | `101.20 MB` ONNX (`c6a9097d...`) | **D-FINE is `22%` smaller** |
| **Locked `val_15` Conf / F1** | `val_15` | `conf = 0.55` (`P=93.55%, R=94.97%, F1=94.25%`) | `conf = 0.50` (**`P=96.46%, R=96.79%, F1=96.62%`**) | Both leak-free on `val_15` |
| **Validation `AP` / `AP50` / `AP75`** | `val_15` (`1,039` imgs) | $60.65\%$ / $95.69\%$ / $69.10\%$ | **$67.84\%$ / $96.82\%$ / $78.74\%$** | `YOLOX-M` converges higher in-domain |
| **Inference Speed (L4 GPU, b=1)** | `test_official` | $34.76\text{ ms}$ ($28.77\text{ FPS}$) | **$17.53\text{ ms}$ ($57.03\text{ FPS}$)** | **`YOLOX-M` is `2.0×` faster** |
| **Overall $\text{mAP}@[0.50:0.95]$** | **Official Test (`5,754` imgs)** | **$48.94\%$** | $44.84\%$ | **`D-FINE-M` (`+4.10%`)** |
| **$\text{AP}_{50}$ (Drone Detection)** | **Official Test (`5,754` imgs)** | **$92.86\%$** | $90.28\%$ | **`D-FINE-M` (`+2.58%`)** |
| **$\text{AP}_{75}$ (Strict Localization)** | **Official Test (`5,754` imgs)** | **$46.14\%$** | $38.12\%$ | **`D-FINE-M` (`+8.02%`)** |
| **$\text{AP}_{\text{small}}$ ($< 32\text{ px}$)** | **Official Test (`5,754` imgs)** | **$46.47\%$** | $42.07\%$ | **`D-FINE-M` (`+4.40%`)** |
| **$\text{AP}_{\text{medium}}$ ($32\text{--}96\text{ px}$)** | **Official Test (`5,754` imgs)** | **$60.84\%$** | $56.51\%$ | **`D-FINE-M` (`+4.33%`)** |
| **$\text{AP}_{\text{large}}$ ($> 96\text{ px}$)** | **Official Test (`5,754` imgs)** | **$70.19\%$** | $60.39\%$ | **`D-FINE-M` (`+9.80%`)** |
| **$\text{AR}@100$ (Max-100 Recall)** | **Official Test (`5,754` imgs)** | **$59.94\%$** | $51.50\%$ | **`D-FINE-M` (`+8.44%`)** |

*(Historical note: During earlier Phase 4/9 exploratory work on Tesla T4 using COCO pre-split `val_half.json` [`3,394` imgs], an initial `D-FINE` run hit `48.2%` mAP / `92.6%` AP50 and an initial `8.94M`-param `YOLOX-S` [`35.85 MB` ONNX] prototype hit `53.44%` mAP / `94.0%` AP50 before upgrading to `YOLOX-M` under the stricter `train_85`/`val_15`/`test_official` zero-leakage protocol.)*

### 12.2 Training Convergence Logs (NVIDIA L4 Production & Historical T4 Runs)

1. **Production D-FINE HGNetv2-M (`logs/dfine_l4_train.log`, NVIDIA L4 24 GB, 15 Epochs, `79.36 min`)**:
   - Initialized from `dfine_m_obj2coco.pth` (`stop_epoch=12`, `640×640`, `total_batch_size=24`, AMP + ModelEMA).
   - Logged `val_15` AP progressed `49.8%` (`90.2%` AP50) $\to$ `56.4%` (`94.0%`) $\to$ `58.3%` (`95.1%`) up to **`best_stg2.pth` (epoch 14)**: `AP = 60.65%`, `AP50 = 95.69%`, `AP75 = 69.10%`, `AP_small = 58.58%`, `AR@100 = 68.45%` (outperforming `best_stg1.pth` ep 11 at `60.24%` AP). Epoch 10 $\to$ 14 training loss converged from `17.9622` to `16.4079` (`loss_bbox = 0.0292`, `loss_giou = 0.2843`).
2. **Production YOLOX-M (`logs/yolox_l4_train.log`, NVIDIA L4 24 GB, 20 Epochs, `47.60 min`)**:
   - Initialized from `yolox_m.pth` (`640×640`, `multiscale_range=2`, `-b 32`, `--fp16`, `no_aug_epochs=5`).
   - Logged `val_15` AP progressed across evaluations: `48.5%` (`94.5%` AP50) $\to$ `52.8%` (`95.3%`) $\to$ `56.1%` (`95.7%`) $\to$ `64.7%` (`96.7%`) $\to$ `66.0%` (`96.8%`) $\to$ `66.6%` (`96.8%`) $\to$ **`67.84%` AP (`96.82%` AP50, `78.74%` AP75, `67.40%` AP_small)** at Epoch 20.
3. **RL Policy (`DroneAgent`) Training (`checkpoints/drone_tracker_policy_best.pth`, `39.8 KB`)**:
   - Verified `17` simultaneous candidates per step (`4` background random clutter + up to `12` real swarm drone hard negatives + `1` ground truth).
   - Benchmarked at `10` epochs in `3.63 s` on Tesla T4; trained full **`600` epochs** with $\epsilon: 1.0 \to 0.05$.
4. **Historical Phase 4/9 Tesla T4 Prototype Runs (Archived Reference)**:
   - Earlier `D-FINE` T4 run (`PID 9677`): total loss `39.11 -> 22.27`, `loss_bbox 0.4775 -> 0.0687`, `loss_giou 1.8226 -> 0.7566`; `val_half` `48.2%` mAP.
   - Earlier `YOLOX-S` (`8.94M`, `12` epochs, `FP16`) T4 run: total loss `4.40 -> 2.70 -> 2.30 -> 2.00`, `val_half` `53.44%` mAP / `94.0%` AP50 (`81.3 FPS` T4).

### 12.3 Tracking Ablation: Static Gate + Greedy vs. Dynamic Gate + Hungarian (`UAVSwarm-02`, 156 Frames)

Evaluated against `results/uavswarm_02_gt.txt` (2,984 annotations, 25 FPS):

| Tracking Metric | Static $250\text{ px}$ Gate + Greedy `argmax` | Dynamic Gate $[20, 65]\text{ px}$ + Hungarian | Improvement |
| :--- | :---: | :---: | :--- |
| **Erratic Jumps ($> 25\text{ px}$)** | $1{,}059$ jumps | **$0$ consecutive jumps** | **100% eliminated** |
| **Mean Frame Displacement** | $18.4\text{ px}$ | **$2.40\text{ px}$** | **$87\%$ smoother** |
| **Median Frame Displacement** | $3.2\text{ px}$ | **$0.40\text{ px}$** | Matches true hovering kinematics |
| **Average Track Purity** | $34.2\%$ | **$100.00\%$** | **$+65.8\%$ (flawless)** |
| **Tracks with 100% Purity (Zero ID switches)** | $2 / 75$ ($2.7\%$) | **$37 / 37$ ($100.0\%$)** | Every track keeps its true drone ID |
| **Max Track Lifetime** | $156$ frames (with jumps) | **$155$ frames (pure, smooth trail)** | Continuous tracing |
| **Ground Truth Detection Recall** | $92.6\%$ | **$95.8\%$** | **$+3.2\%$** |

**Top 10 Persistent Ground-Truth Mapped Tracks (`UAVSwarm-02` Dynamic Gate + Hungarian)**:
- Track `#16`: $155$ frames continuous ($100.0\%$ purity $\to$ GT `#3`)
- Track `#7`: $154$ frames continuous ($100.0\%$ purity $\to$ GT `#12`)
- Track `#15`: $148$ frames continuous ($100.0\%$ purity $\to$ GT `#1`)
- Track `#13`: $146$ frames continuous ($100.0\%$ purity $\to$ GT `#15`)
- Track `#31`: $130$ frames continuous ($100.0\%$ purity $\to$ GT `#11`)
- Track `#6`: $120$ frames continuous ($100.0\%$ purity $\to$ GT `#16`)
- Track `#5`: $110$ frames continuous ($100.0\%$ purity $\to$ GT `#17`)
- Track `#28`: $100$ frames continuous ($100.0\%$ purity $\to$ GT `#17`)
- Track `#17`: $96$ frames continuous ($100.0\%$ purity $\to$ GT `#10`)
- Track `#23`: $91$ frames continuous ($100.0\%$ purity $\to$ GT `#14`)

*(Note: By comparison, the earlier Phase 5 static $250\text{ px}$ gate baseline ran at $6.8\text{ FPS}$ end-to-end on T4 ($22.95\text{ s}$ total), tracing $21\text{--}28$ active drones/frame and $3{,}675$ points across $75$ tracks with 75th-percentile track length $123.5$ frames—where top IDs 1, 2, 10, 11, 12, 13, 15, 17, 18, 19 lasted 156 frames only because ID jumping kept tracks artificially alive across $1{,}059$ jumps.)*

### 12.4 Full 36-Sequence MOT Benchmark: Tier A (Oracle) & Tier B (End-to-End)

Using `evaluations/mot_metrics.py` (`MOTA`, `IDF1`, and 19-threshold `HOTA`), we evaluated all **36 official UAVSwarm test sequences (`5,754` frames, `325` GT tracks, `37,947` MOT GT boxes)** under both **Tier A** (`evaluations/evaluate_tracking.py` — shuffled GT boxes, zero ID leakage) and **Tier B** (`evaluations/evaluate_end_to_end.py` — L4 `D-FINE-M conf=0.55` and `YOLOX-M conf=0.50` detections) ([evaluations/results.md](evaluations/results.md)):

| Tier / Detector Setup | Tracker Method | MOTA (%) ↑ | HOTA (%) ↑ | DetA (%) ↑ | AssA (%) ↑ | IDF1 (%) ↑ | IDSW ↓ | Avg Purity (%) ↑ | CPU FPS (Assoc) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier A (`Oracle GT`, 36 Seqs)** | **`DroneRLTracker` (Ours)** | **99.84** | **97.14** | 99.96 | **94.41** | **96.12** | 44 | **99.32** | 1,599.5 |
| **Tier A (`Oracle GT`, 36 Seqs)** | `DynGate_Hungarian_NoRL` (`dist`) | **99.84** | 97.13 | 99.96 | 94.39 | **96.12** | **42** | **99.32** | 3,631.1 |
| **Tier A (`Oracle GT`, 36 Seqs)** | `Kalman_IoU_Hungarian` (SORT) | 99.00 | 94.33 | 99.52 | 89.41 | 92.31 | 197 | 99.03 | 3,462.6 |
| **Tier A (`Oracle GT`, 36 Seqs)** | `Static250_GreedyRL` (Shrimp) | 99.19 | 92.96 | **100.00** | 86.41 | 91.19 | 307 | 91.97 | 1,559.0 |
| **Tier B (`D-FINE-M`, `conf=0.55`)** | **`DroneRLTracker` (Ours)** | **82.87** | **68.12** | 64.93 | **72.52** | **86.91** | **96** | **99.43** | 1,735.1 *(+34.8ms L4)* |
| **Tier B (`D-FINE-M`, `conf=0.55`)** | `DynGate_Hungarian_NoRL` | 82.86 | 68.09 | 64.93 | 72.48 | 86.88 | 99 | **99.43** | 3,710.0 |
| **Tier B (`D-FINE-M`, `conf=0.55`)** | `Kalman_IoU_Hungarian` (SORT) | 82.42 | 67.03 | 64.81 | 70.44 | 84.76 | 209 | 99.09 | **4,225.8** |
| **Tier B (`D-FINE-M`, `conf=0.55`)** | `Static250_GreedyRL` (Shrimp) | 80.87 | 62.05 | **64.96** | 60.52 | 75.51 | 861 | 83.61 | 1,706.7 |
| **Tier B (`YOLOX-M`, `conf=0.50`)** | **`DroneRLTracker` (Ours)** | **81.94** | **66.85** | 62.84 | **72.52** | **87.18** | **54** | **99.42** | 1,766.8 *(+17.5ms L4)* |
| **Tier B (`YOLOX-M`, `conf=0.50`)** | `DynGate_Hungarian_NoRL` | **81.94** | **66.85** | 62.84 | **72.52** | **87.18** | **54** | **99.42** | 3,296.1 |
| **Tier B (`YOLOX-M`, `conf=0.50`)** | `Kalman_IoU_Hungarian` (SORT) | 81.37 | 65.65 | 62.65 | 70.23 | 84.61 | 170 | 99.18 | **3,873.7** |
| **Tier B (`YOLOX-M`, `conf=0.50`)** | `Static250_GreedyRL` (Shrimp) | 80.62 | 60.87 | **62.88** | 60.49 | 77.21 | 569 | 88.14 | 1,586.3 |

- **Diagnostic Subsets (`DroneRLTracker`)**:
  - **Local 4 (`UAVSwarm-02, 04, 08, 14`)**: Tier A = `99.99% MOTA / 99.88% HOTA / 99.84% IDF1 / 1 IDSW / 100.00% Purity`; Tier B `D-FINE-M` = `90.73% / 68.95% / 94.60% / 11`; Tier B `YOLOX-M` = `91.32% / 68.62% / 95.22% / 3`.
  - **Literature Subset (`UAVSwarm-06, 28, 30, 46`)**: Tier A = `99.58% MOTA / 94.09% HOTA / 92.96% IDF1 / 20 IDSW`; Tier B `D-FINE-M` = `79.52% / 62.53% / 77.68% / 37`; Tier B `YOLOX-M` = `59.59% / 60.27% / 73.86% / 19`.
- **Key Tier-A Findings (`23/36` sequences hit `100.00%`, `31/36 > 92%` HOTA)**:
  1. Dynamic gating + Hungarian cuts Tier-A ID switches by **`7.0×` (`307 -> 44`)** vs. `Static250_GreedyRL`, and cuts Tier-B ID switches by **`2.2×–3.1×` over SORT (`209 -> 96`, `170 -> 54`)** and **`9×–10.5×` over `Static250_GreedyRL` (`861 -> 96`, `569 -> 54`)**.
  2. **Root Causes of the 44 Tier-A IDSW & 17 FN**: *(a)* Long GT occlusion gaps exceeding `MAX_MISSED = 8` (`UAVSwarm-28, 38, 46, 70, 72` have `9–43` frame gaps); *(b)* raw-pixel `[20, 65] px` gate overflow on high-resolution/fast sequences (`UAVSwarm-44` `1919×870`, `UAVSwarm-52` `max_speed = 64.07 px/f`); and *(c)* training at fixed `gate_radius=250.0` vs. `[20, 65] px` inference scaling.

### 12.5 Hungarian Cost-Matrix Formulation Ablation (`evaluations/testing_different_stratergies/`)

We evaluated five alternative in-gate (`dist <= dyn_gate`) Hungarian cost functions across all **36 official test sequences (`5,754` frames, `37,947` GT boxes)** in [evaluations/testing_different_stratergies/](evaluations/testing_different_stratergies/) and [evaluations/results.md](evaluations/results.md):

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

- **Mathematical Takeaways**:
  1. **Empirical Q-Value Band**: Across `37k–39k` gated candidate evaluations per mode, `DroneQNetwork` outputs tightly clustered positive discounted returns (`Q ∈ [16.33, 19.62]`, mean `~19.07`, std `~0.56`).
  2. **Why `dist * (1.5 - tanh((Q - 17.5)/2.5))` Is Best Overall**: Centering `Q` via `tanh` expands that narrow positive band into a bounded positive weight `(0.5, 2.5)` while keeping Euclidean distance strictly positive — placing **#1 in HOTA on all 3 splits** (`97.14%`, `68.12%`, `66.85%`) and #1 on every `D-FINE-M` metric (`96` IDSW).
  3. **Why `d / q` Mirrors Pure `dist`**: Because `1/Q` varies by only ~`3%` across `Q ∈ [16.33, 19.62]`, dividing `d` by raw `q` leaves Hungarian assignments almost untouched (`YOLOX-M` matches pure `dist` on `36/36` sequences).
  4. **Why `-q` and Negative Distance-Scaled Forms Degrade**: Minimizing `-q` alone discards sub-gate spatial precision, increasing IDSW by `+50%–75%` (`77` / `167` / `81`). Negative distance-scaled forms (`-d * q` and `-d / q`, evaluated and omitted from retained runners) prefer the farthest in-gate candidate under minimization (`2,853–6,007` IDSW). Conversely, `-q / (d + 1.0)` aligns both terms toward smaller cost and trails `DroneRLTracker` by only `0.00%–0.05%` HOTA.

### 12.6 Literature Standing Across 14 UAVSwarm Papers (2022–2026)

Full protocol provenance, per-paper summaries, and all 14 citations are maintained in [comparision.md](comparision.md):

| Protocol Tier | Paper / System | MOTA (%) ↑ | IDF1 (%) ↑ | HOTA (%) ↑ | IDSW ↓ | Key Context / Comparison |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier A (Oracle GT)** | **`DroneRLTracker` (Ours)** | **99.84** | **96.12** | **97.14** | 44 | **`UAVSwarm-28`**: **`99.08%` MOTA / `83.73%` IDF1 / `8` IDSW** (`1,599.5 FPS` CPU) |
| **Tier A (Oracle GT)** | `GNMOT` *(Wang et al., 2022)* | — | — | — | — | `25/36` perfect vs our `23/36`; `UAVSwarm-28`: `98.4%` MOTA / `76.9%` IDF1 / `14` IDSW |
| **Tier B (UAVSwarm Test)** | `RoeNet` *(Jul 2026)* `[Paywalled ESWA]` | **85.99** | 84.03 | 72.25 | — | Physics-inspired JDT (`96.82%` AP50); both our Tier-B trackers lead in `IDF1` |
| **Tier B (UAVSwarm Test)** | `EECTracker` *(Sep 2026, `YOLOX-X 1088`)* | 84.71 | **90.19** | 69.16 | **54** | `22.9 FPS` (RTX 3090); beats `EECTracker-Light` (`78.73% / 65.60%`) |
| **Tier B (UAVSwarm Test)** | **`D-FINE-M (640) + DroneRLTracker` (Ours)** | **82.87** | 86.91 | **68.12** | 96 | `28.8 FPS` L4 det + `1,735 FPS` CPU trk; beats `SCT-MOT` & `10` others on MOTA |
| **Tier B (UAVSwarm Test)** | **`YOLOX-M (640) + DroneRLTracker` (Ours)** | 81.94 | **87.18** | 66.85 | **54** | **`57.0 FPS` L4 det** + `1,767 FPS` CPU trk; ties `EECTracker`'s `54` IDSW |
| **Tier B (UAVSwarm Test)** | `SCT-MOT` *(Apr 2026, `YOLOX-X 1088`)* | 81.90 | 88.45 | 68.56 | 56 | `22.3 FPS` (RTX 3090); `+0.44%` HOTA edge via 8-frame `SMTP` |
| **Tier B (UAVSwarm Test)** | `KRAfT` *(ICPR 2026)* `[Paywalled]` | — | — | **80.37** | — | Kalman Residual Diffusion (`70.59%` cross-weather `W2C`) |
| **Tier B (UAVSwarm Test)** | `HOMATracker` / `Hybrid-SORT` / `OC-SORT` / `ByteTrack` | 79.2 / 77.4 / 75.7 / 65.0 | 87.1 / 80.1 / 81.8 / 76.5 | 67.0 / 62.8 / 64.7 / 60.6 | 58–709 | Our `640×640` Tier-B trackers outperform all on `MOTA` & `IDF1` |

### 12.7 Local 4-Sequence Runtime Telemetry (`results/`)

#### A. Headline Local Telemetry (`README.md`, `run2.py`)

| Sequence | Resolution | Frames | Actual Swarm Size | Mean Active Swarm (Peak) | Persistent Tracks | Coordinates Traced |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UAVSwarm-02** | $812 \times 428$ | $156$ | $20$ drones | $19.0$ / frame (Peak: $20$) | $20$ | **$2{,}983$ points** |
| **UAVSwarm-04** | $812 \times 428$ | $161$ | $22$ drones | $18.7$ / frame (Peak: $22$) | $22$ | **$3{,}062$ points** |
| **UAVSwarm-08** | $446 \times 270$ | $115$ | $12$ drones | $9.8$ / frame (Peak: $12$) | $17$ | **$1{,}148$ points** |
| **UAVSwarm-14** | $812 \times 428$ | $115$ | $20$ drones | $16.6$ / frame (Peak: $20$) | $22$ | **$1{,}936$ points** |

#### B. Verified Per-Pipeline & Smoke-Test Measurements (`DEVLOG.md`)

| Pipeline / Mode | Sequence | FPS / Latency | Persistent Tracks | Points Traced | Mean Swarm (Peak) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Post-L4 Headless `run.py` (`D-FINE-M`, `conf=0.55`, MPS)** | `UAVSwarm-08` | $2.0\text{--}3.0\text{ FPS}$ (MPS) | $12$ | $996$ | — |
| **Post-L4 Headless `run3.py --mode runtime` (`YOLOX-M`, `conf=0.50`, CPU ONNX)** | `UAVSwarm-08` | $1.4\text{--}1.6\text{ FPS}$ (CPU) | $13$ | $1{,}035$ | — |
| **Phase 7 `run.py` (Precomputed D-FINE dets)** | `UAVSwarm-02 / 04 / 08 / 14` | **$33.5\text{ FPS}$ avg** | $71 / 20 / 11 / 20$ | **$9{,}216$ total** | — |
| **Pipeline 2 `run2.py` (Shuffled GT Oracle)** | `UAVSwarm-02` / `UAVSwarm-04` | $36.1$ / $37.1\text{ FPS}$ | $20$ / $20$ | $2{,}983$ / $3{,}034$ | — |
| **Historical Phase 9 `run3.py` (`YOLOX-S`, `04/08/14`)** | `UAVSwarm-04 / 08 / 14` | Live ONNX Runtime | — | $3{,}062 / 1{,}148 / 1{,}936$ | $18.7(22) / 9.8(12) / 16.6(20)$ |

---

## 13. Complete Phase-by-Phase Engineering & Decision Log (Phases 0–13)

Every milestone from `PLAN.md` and `DEVLOG.md` is captured below in chronological order:

1. **Phase 0 — Scoping, UAVSwarm Analysis & `Shrimp` Architecture Reverse-Engineering (2026-11-02)**:
   - Selected `UAVSwarm-dataset` (`12,598` images, `13` scenes — sky, urban, forest, rural, indoor — `19+` UAV types, `3–23` drones/sequence, MOT16 ground truth).
   - Identified small-object challenge ($<32\times 32\text{ px}$ drones) $\implies$ opted for $800\times 800$ multi-scale training and monitoring $\text{AP}_s$.
   - Selected **D-FINE** (2024–2025 SOTA real-time DETR with FDR) over standard YOLO baselines initially, plus Google Colab Tesla T4 (`15–16 GB` VRAM) via `google-colab-cli`.
   - Analyzed `SamratRay2005/Shrimp` source code and mapped all parameters (`gate_radius`, clutter, `QNetwork`, buffer, epochs, cleaning thresholds).
2. **Phase 1 — Dataset Pipeline (2026-11-02)**:
   - Cloned `UAVSwarm-dataset` on Colab (`/content/uavswarm`, `7,096` train / `5,862` test images) and discovered pre-built COCO files (`train.json` 8.3 MB, `test.json` 951 KB, `train_half.json` 4.2 MB / 3,450 train imgs / 25,396 annotations, `val_half.json` 4.0 MB / 3,394 val imgs / 24,265 annotations).
   - Built `dataset/download_uavswarm.py` and `dataset/mot2coco.py`.
   - Ran `shrimp_drone/mot_to_csv.py` across 36 training sequences: `355` raw tracks / `49,661` rows $\to$ **`323` cleaned tracks / `47,011` trajectory points**.
3. **Phase 2 — D-FINE Configuration (`configs/dfine/custom/dfine_hgnetv2_m_uavswarm.yml`)**:
   - Configured `num_classes: 1`, `remap_mscoco_category: False`, `HGNetv2-B2` (`return_idx: [1, 2, 3]`, `use_lab: True`, `pretrained: False`), `HybridEncoder` (`in_channels: [384, 768, 1536]`, `hidden_dim: 256`, `depth_mult: 0.67`), `DFINETransformer` (`num_layers: 4`), AMP enabled, `total_batch_size: 4`, `800×800` val / `[640..800]` random resize train, H+V flip (`p=0.3` vertical), `RandomPhotometricDistort`, `RandomZoomOut`, `RandomIoUCrop`, `AdamW` (`lr=5e-5`, backbone `1e-5`), `MultiStepLR` (`[55, 68]`).
4. **Phase 3 — Shrimp $\to$ Drone RL Adaptation**:
   - Built `shrimp_drone/drone_env.py` (`DroneTrackingEnv` with `get_dynamic_gate()` + 12 real swarm hard negatives + 4 random clutter), `shrimp_drone/train_drone_agent.py` (`DroneQNetwork` 5→128→64→1, 50k buffer, 600 epochs, `+0.5` near-miss reward), and `shrimp_drone/run_drone_tracker.py` (150-frame history rendering & CSV/JSON export).
5. **Phase 4 — Cloud Training & Backups**:
   - Downloaded `dfine_m_coco.pth` (`79.1 MB`) for transfer learning and trained an initial D-FINE prototype on Tesla T4 (`val_half` `AP50 = 92.6%`, `AR50 = 96.0%`, `mAP = 48.2%`).
   - Trained RL policy for 600 epochs $\to$ `checkpoints/drone_tracker_policy_best.pth` (`39.8 KB`).
   - Shifted operational objective from Shrimp line-crossing counting (`VectorHysteresisCounter`) to **continuous spatial trajectory tracing**.
6. **Phase 5 — Image-to-Video Compilation & Initial Tracking**:
   - Built `shrimp_drone/test_video_tracking.py`, compiled `test/UAVSwarm-02` (`156` frames, `812×428` @ 25 FPS) into `results/uavswarm_test_raw.mp4` (`788 KB`).
   - Verified end-to-end inference, uncovering the static-gate jumping problem ($1{,}059$ jumps).
7. **Phase 6 — Velocity-Adaptive Dynamic Gating & Hungarian Stabilization**:
   - Based on user feedback (*"why not use a dynamic gating based on the velocity of the drones in the video"*), formulated $R_{\text{dynamic}} = \mathrm{clip}(20 + 2\|\mathbf{v}\| + 4k, 20, 65)$ and global Hungarian matching $C_{i,j} = d_{i,j}(1.5 - \tilde{Q}(\mathbf{s}_{i,j}))$ with $\tilde{Q} = \tanh((Q - 17.5)/2.5)$.
   - Eliminated erratic jumps ($1{,}059 \to 0$) and reached **$100.0\%$ track purity** ($37/37$ tracks). Created `scripts/render_clean_video.py`.
8. **Phase 7 — Unified Root `run.py`, Multi-Sequence Suite & Live MPS D-FINE**:
   - Created `run.py` mirroring `Shrimp/run.py`, compiled `UAVSwarm-04` (`907 KB`, 161f), `UAVSwarm-08` (`147 KB`, 115f), `UAVSwarm-14` (`361 KB`, 115f).
   - Standardized clean architecture (removed synthetic placeholders like `Drone/annotator.py`), preserved research assets for paper publication, and verified live D-FINE runtime inference on Apple Silicon MPS (`run.py`, `UAVSwarm-08`, `1.6 FPS`, `14` tracks).
9. **Phase 8 — Ground Truth Oracle Pipeline (`run2.py`) (2026-11-03)**:
   - Implemented `run2.py` feeding isolated GT boxes (stripped of IDs for association and `random.shuffle`-d per frame) to prove the upper-bound tracker performance into `result_without_custom_object_detection/`.
10. **Phase 9 — Initial `YOLOX-S` Prototype & Dual-Mode `run3.py` (2026-11-03)**:
    - Provisioned Colab `yolox-trainer` (Tesla T4, 15.6 GB VRAM), cloned full `3.7 GB` `UAVSwarm-dataset` (`12,963` images across 72 sequences), fine-tuned an initial single-class `YOLOX-S` (`8.94M` params) on `train_half.json` (`3,450` imgs) / `val_half.json` (`3,394` imgs) for 12 epochs (`FP16`, `batch=16`), reaching `53.44%` mAP / `94.0%` AP50 on `val_half` (`81.3 FPS` T4, `35.85 MB` ONNX), and built `run3.py` (`--mode runtime` & `--mode benchmark`) outputting to `result_with_yolox/`.
11. **Phase 10 — Multi-Video Validation, Workspace Hygiene & Publication Docs (2026-11-03)**:
    - Validated `run3.py --mode runtime` across `04`, `08`, and `14`.
    - Purged temporary scratch scripts from `scripts/`, retained only clean benchmark inputs in `results/`, documented YOLOX training in `notebooks/04_yolox_finetune.py`, updated `README.md` / `PLAN.md` / `DEVLOG.md`, and committed weights via Git LFS (`drone-swarm-tracking`).
12. **Phase 11 — Full 36-Sequence Tier-A Benchmark Suite & 14-Paper Literature Analysis (2026-11-20)**:
    - Implemented unit-tested `MOTA`, `IDF1`, and 19-threshold `HOTA` evaluators in `evaluations/mot_metrics.py` and full 36-sequence ablation runner `evaluations/evaluate_tracking.py` (`5,754` frames, `325` tracks, `37,947` boxes), achieving **`99.84%` MOTA / `97.14%` HOTA / `96.12%` IDF1 / `44` IDSW / `1,599.5 FPS`**.
    - Authored `evaluations/results.md` and `comparision.md` across 14 published UAVSwarm studies (2022–2026), directly contrasting Tier-A tracking against `GNMOT` (`UAVSwarm-28`: `99.08%` vs `98.4%` MOTA).
13. **Phase 12 — Zero-Leakage NVIDIA L4 Retraining (`D-FINE-M` & `YOLOX-M`) & Tier-B End-to-End Evaluation (2026-11-20)**:
    - Constructed chronological `train_85` (`5,805` imgs) / `val_15` (`1,039` imgs) splits and provisioned two parallel **NVIDIA L4 (24 GB)** Colab VMs (`dfine-gpu`, `yolox-gpu`; A100 returned HTTP 503).
    - Trained `D-FINE HGNetv2-M` (`15` ep, `79.36 min`, `best_stg2.pth` ep 14 -> **`78.57 MB` EMA**, MD5 `cba23409f653b79e347db3c9430a338b`) and upgraded `YOLOX` to **`YOLOX-M`** (`20` ep, `47.60 min` -> **`101.20 MB` ONNX**, MD5 `c6a9097d1f490a912c8fa62c774589fe`), locking `val_15` thresholds (`0.55` / `0.50`).
    - Evaluated once on all `5,754` unseen `test_official` frames (Tier D: `92.86%` / `90.28%` AP50; Tier B via `evaluate_end_to_end.py`: `82.87% MOTA / 68.12% HOTA / 86.91% IDF1 / 96 IDSW` and `81.94% / 66.85% / 87.18% / 54 IDSW`).
    - Updated `run.py` (`SCORE_THRES = 0.55`), `run3.py` (shape auto-detection + `conf_thresh = 0.50`), and `dfine_core` (lazy `SummaryWriter`), verified headless smoke tests on `UAVSwarm-08`, and terminated both Colab VMs cleanly (`0.00/hr` CCU).
14. **Phase 13 — Hungarian Cost-Matrix Strategy Ablations & `evaluations/` Organization (2026-11-20)**:
    - Implemented standalone strategy ablations in `evaluations/testing_different_stratergies/` (`evaluate_neg_q.py`, `evaluate_neg_q_over_d.py`, `evaluate_d_over_q.py`), confirming `dist * (1.5 - tanh((Q - 17.5)/2.5))` achieves #1 HOTA across Tier A and both Tier B splits.
    - Organized `evaluations/` cleanly into `data/` (`uavswarm_test_gt/`, `detections_dfine/`, `detections_yolox/`), `results/` (`tier_a/`, `tier_b/`, `predictions/`, `plots/`), and `testing_different_stratergies/`, updating all runner scripts and documentation.

---

## 14. Critical Bugs, Cloud Hurdles & Engineering Solutions

A quick reference of every nontrivial bug encountered in `DEVLOG.md` and how each was solved:

| # | Symptom / Failure | Root Cause | Exact Fix Implemented |
| :---: | :--- | :--- | :--- |
| **1** | D-FINE cross-entropy target out-of-bounds crash | Raw UAVSwarm JSON used `category_id: 1` (`'pedestrian'`), while `num_classes: 1` + `remap_mscoco_category: False` expects `0`-indexed targets `[0]`. | Sanitized `train.json`, `val.json`, `full_train.json` (and `train_85` / `val_15`) to `category_id: 0`, name `'drone'`. |
| **2** | D-FINE YAML dataloader assertion error | Base `include/dataloader.yml` sets `total_batch_size`; specifying `batch_size` simultaneously triggers an exclusivity assertion. | Overrode `total_batch_size` (instead of `batch_size`) in D-FINE config (`4` on T4; `24` on L4). |
| **3** | D-FINE runtime tensor shape error on arbitrary video sizes & offline stall | `HybridEncoder` uses a fixed `pos_embed` of `400` tokens ($20\times 20$ grid at stride 32 $\implies 640\times 640$). Default `pretrained: True` also tries downloading backbone weights. | Forced `T.Resize((640, 640))` in `DFineDetector` and set `pretrained: False` in `dfine_hgnetv2_m_uavswarm.yml`. |
| **4** | `colab download` socket drop on `313 MB` `best_stg1.pth` | Jupyter server contents API encodes files as base64 in a single JSON HTTP payload (`GET /api/contents/...`), stalling/closing Tornado buffers above ~`50 MB`. | Extracted EMA-only weights (`dfine_drone_weights.pth`), ran `split -b 15M` into chunks, downloaded chunks, reassembled locally via `cat`, and mirrored via `scripts/sync_backups.py`. |
| **5** | Colab long-running jobs timing out | `google-colab-cli` (`colab exec`) defaults to a `30.0 s` client timeout, and Colab VMs disconnect on idle. | Passed `--timeout 3600`, launched detached `nohup` jobs writing to `/content/*.log`, and mirrored continuously via `sync_backups.py --loop`. |
| **6** | Megvii YOLOX install failures on Colab Python 3.13 | `setup.py` asserted `torch` wasn't found inside isolated PEP-517 build env, and `onnx-simplifier` failed metadata build on Python 3.13. | Stripped `onnx-simplifier` from `requirements.txt` and installed with `pip install --no-build-isolation -e .`. |
| **7** | PyTorch Dynamo ONNX export broken for YOLOX | Modern PyTorch Dynamo-based ONNX export produced incomplete external references on YOLOX's head. | Used TorchScript tracing (`torch.onnx.export(..., dynamo=False, opset_version=14)`) + `weights_only=False` checkpoint load, producing self-contained `yolox_drone_finetuned.onnx` (`101.20 MB` for `YOLOX-M`, shape `[1, 8400, 6]`; earlier `35.85 MB` for `YOLOX-S`). |
| **8** | `DroneAgent.__init__()` unexpected keyword `epochs` | Training script passed `epochs` to `DroneAgent` constructor instead of only the training loop. | Removed `epochs` argument from `DroneAgent.__init__` in `shrimp_drone/train_drone_agent.py`. |
| **9** | `run.py` `AttributeError: np.complex_` on import under NumPy 2.x | `dfine_core/src/core/_config.py` and `dfine_core/src/solver/det_engine.py` eagerly imported `torch.utils.tensorboard.SummaryWriter`, triggering an incompatible local TensorFlow installation. | Replaced eager imports with `TYPE_CHECKING` guard and lazy `SummaryWriter` import inside `BaseConfig.writer`; headless `run.py` MPS inference verified clean. |
| **10** | Colab A100 provisioning returned HTTP 503 | Transient A100 GPU unavailability during final detector retraining. | Provisioned two parallel **NVIDIA L4 (24 GB)** VMs (`dfine-gpu` & `yolox-gpu`), training both models concurrently (`79.36 min` & `47.60 min`) and terminating cleanly afterwards. |

---

## 15. Output Data Schemas, GUI Controls & CLI Reference

### 15.1 Installation & Dependencies

```bash
git clone https://github.com/SamratRay2005/Shrimp.git
cd Shrimp
pip install torch torchvision opencv-python onnxruntime numpy pandas scipy pillow pyyaml loguru
```

### 15.2 Complete CLI Commands for All Three Pipelines

```bash
# ── Pipeline 1: Live D-FINE Transformer + RL Tracker (result_with_our_dfine/) ──
python run.py
python run.py --video results/uavswarm_test_04_raw.mp4
python run.py --video results/uavswarm_test_08_raw.mp4
python run.py --video results/uavswarm_test_14_raw.mp4
python run.py --video results/uavswarm_test_04_raw.mp4 --headless
# Optional: --weights, --policy, --conf (default 0.55), --benchmark-gt

# ── Pipeline 2: Shuffled Ground-Truth Oracle + RL Tracker (result_without_custom_object_detection/) ──
python run2.py
python run2.py --video results/uavswarm_test_04_raw.mp4
python run2.py --video results/uavswarm_test_08_raw.mp4
python run2.py --video results/uavswarm_test_14_raw.mp4
python run2.py --video results/uavswarm_test_04_raw.mp4 --headless
# Optional: --gt path/to/gt.txt, --policy

# ── Pipeline 3A: Live YOLOX-M ONNX + RL Tracker (result_with_yolox/) ──
python run3.py --mode runtime
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_08_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_14_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode runtime --headless

# ── Pipeline 3B: Official UAVSwarm Benchmark YOLOX Detections + RL Tracker ──
python run3.py --mode benchmark
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode benchmark
# Optional: --weights, --det path/to/det.txt, --policy, --conf (default 0.50), --output, --csv, --json

# ── 36-Sequence Benchmark Suite & Cost-Strategy Ablations (evaluations/) ──
python3 evaluations/mot_metrics.py
python3 evaluations/evaluate_tracking.py
python3 evaluations/evaluate_end_to_end.py
python3 evaluations/testing_different_stratergies/evaluate_neg_q.py
python3 evaluations/testing_different_stratergies/evaluate_neg_q_over_d.py
python3 evaluations/testing_different_stratergies/evaluate_d_over_q.py

# ── Re-train the RL Association Agent from cleaned trajectory CSV ──
python shrimp_drone/train_drone_agent.py \
    --csv drone_training_data.csv \
    --epochs 600 \
    --save_path checkpoints/drone_tracker_policy.pth
```

### 15.3 Interactive OpenCV GUI Controls

When running without `--headless`, a resizable window displays bounding boxes, track ID badges with live dynamic gate radius, neon trails, dynamic gate rings, and telemetry HUD:
- **`SPACE`**: Pause and resume video playback frame by frame.
- **`q`** or **`ESC`**: Cleanly terminate tracking and write the output `.mp4`, `.csv`, and `.json` files.

### 15.4 Standardized Output Data Formats

Each pipeline exports a CSV and JSON trajectory file in its respective output folder:

**CSV Format (`trajectories_<video>.csv`)** (`run2.py` also includes a `gt_id` column):
```csv
track_id,frame,cx,cy,vx,vy,x1,y1,x2,y2
1,1,278.76,124.65,0.0,0.0,260.87,110.35,296.65,138.96
1,2,278.28,125.90,-0.48,1.25,260.09,110.96,296.47,140.84
```

**JSON Format (`trajectories_<video>.json`)**:
```json
{
  "1": [
    {
      "frame": 1,
      "cx": 278.76,
      "cy": 124.65,
      "vx": 0.0,
      "vy": 0.0,
      "box": [260.87, 110.35, 296.65, 138.96]
    }
  ]
}
```

---

## 16. Repository File Map, Tech Stack & Citations

### 16.1 Complete Directory Tree & File Reference

```
RM_Shrimp_extension/
│
├── run.py                           # Pipeline 1: Live D-FINE-M detection + RL tracking
├── run2.py                          # Pipeline 2: Ground truth detection (shuffled) + RL tracking
├── run3.py                          # Pipeline 3: Live YOLOX-M ONNX / benchmark det.txt + RL tracking
├── study.md                         # This complete first-principles guide & reference
├── PLAN.md                          # Phase 0-10 roadmap & architecture comparison
├── DEVLOG.md                        # Chronological engineering decision & debugging log
├── README.md                        # Project overview, benchmark tables, and CLI quickstart
├── comparision.md                   # 14-paper (2022–2026) UAVSwarm literature benchmark
├── run.txt                          # Verification command & output log
│
├── checkpoints/                     # Trained detector & RL policy weights (Git LFS)
│   ├── dfine_drone_weights.pth      # Fine-tuned D-FINE-M clean EMA weights (78.57 MB)
│   ├── yolox_drone_finetuned.onnx   # Fine-tuned YOLOX-M single-class ONNX (101.20 MB, 25.28M params)
│   ├── yolox_s.onnx                 # Pretrained COCO YOLOX-S fallback baseline (35.86 MB)
│   ├── drone_tracker_policy_best.pth# Trained 5→128→64→1 RL association policy (39.8 KB)
│   └── test_drone_policy_best.pth   # Backup RL policy checkpoint (39.8 KB)
│
├── result_with_our_dfine/           # Outputs from Pipeline 1 (D-FINE-M + RL)
├── result_without_custom_object_detection/ # Outputs from Pipeline 2 (Shuffled GT + RL)
├── result_with_yolox/               # Outputs from Pipeline 3 (YOLOX-M + RL)
│
├── results/                         # Clean raw benchmark videos & official annotations
│   ├── uavswarm_test_raw.mp4        # UAVSwarm-02 raw video (156 frames, 812x428)
│   ├── uavswarm_test_04_raw.mp4     # UAVSwarm-04 raw video (161 frames, 812x428)
│   ├── uavswarm_test_08_raw.mp4     # UAVSwarm-08 raw video (115 frames, 446x270)
│   ├── uavswarm_test_14_raw.mp4     # UAVSwarm-14 raw video (115 frames, 812x428)
│   ├── uavswarm_*_gt.txt            # Official MOT16 ground-truth files (02, 04, 08, 14)
│   └── uavswarm_*_yolox_det.txt     # Official benchmark YOLOX detections (02, 04, 08, 14)
│
├── evaluations/                     # Organized 36-sequence benchmarks & strategy ablations
│   ├── mot_metrics.py               # Unit-tested MOTA, IDF1, and 19-threshold HOTA evaluators
│   ├── evaluate_tracking.py         # Tier-A 36-sequence oracle-box benchmark & ablations
│   ├── evaluate_end_to_end.py       # Tier-B 36-sequence D-FINE-M / YOLOX-M end-to-end benchmark
│   ├── results.md                   # Complete Tier-A/D/B & cost-matrix benchmark report
│   ├── data/                        # uavswarm_test_gt/, detections_dfine/, detections_yolox/
│   ├── results/                     # tier_a/, tier_b/, predictions/, plots/
│   └── testing_different_stratergies/
│       ├── evaluate_neg_q.py        # Cost strategy ablation: -q
│       ├── evaluate_neg_q_over_d.py # Cost strategy ablation: -q/(d+1e-3) & -q/(d+1.0)
│       ├── evaluate_d_over_q.py     # Cost strategy ablation: d/q
│       └── results/                 # Strategy CSV & JSON comparisons (12 result files)
│
├── shrimp_drone/                    # Core RL environment, DQN trainer & dataset utilities
│   ├── drone_env.py                 # DroneTrackingEnv (dynamic gating + swarm clutter hard negatives)
│   ├── train_drone_agent.py         # DroneQNetwork + ReplayBuffer + DroneAgent (DQN training)
│   ├── mot_to_csv.py                # MOT16 → cleaned continuous trajectory CSV converter
│   ├── run_drone_tracker.py         # Modular trajectory tracing inference script
│   ├── test_video_tracking.py       # Sequence-to-MP4 compilation & tracking evaluation
│   ├── train_dfine_runner.py        # Colab D-FINE training launcher
│   └── train_rl_runner.py           # Colab RL training launcher
│
├── dataset/                         # Dataset preparation utilities
│   ├── download_uavswarm.py         # Automated UAVSwarm downloader & stats validator
│   └── mot2coco.py                  # Sequence-split MOT16 → COCO JSON converter
│
├── dfine_core/                      # Vendored D-FINE architecture, YAMLConfig & configs
│   └── configs/dfine/custom/dfine_hgnetv2_m_uavswarm.yml
├── dfine_finetune/                  # Custom D-FINE training YAML configurations
│
├── notebooks/                       # Reproducible Colab GPU training pipelines
│   ├── 01_dataset_preparation.py    # UAVSwarm MOT → COCO + trajectory CSV
│   ├── 02_dfine_finetune.py         # D-FINE HGNetv2-M fine-tuning & ONNX export
│   ├── 03_rl_agent_training.py      # 600-epoch RL Q-Network agent training
│   └── 04_yolox_finetune.py         # YOLOX fine-tuning & ONNX export
│
├── logs/                            # Training terminal logs & summaries (L4 & T4/RL runs)
│   ├── dfine_l4_train.log           # NVIDIA L4 D-FINE HGNetv2-M training log (15 ep)
│   ├── yolox_l4_train.log           # NVIDIA L4 YOLOX-M training log (20 ep)
│   └── dfine_train.log / rl_train.log
└── scripts/
    └── sync_backups.py              # One-shot / --loop Colab backup mirror script
```

### 16.2 Core Stack & Citations

- **Stack**: Python 3.10+, PyTorch 2.0+, ONNX Runtime, OpenCV, NumPy, Pandas, SciPy (`linear_sum_assignment`), Pillow, PyYAML, Loguru.
- **UAVSwarm Dataset**: [A Benchmark for Multi-UAV Tracking in Complex Aerial Scenarios](https://github.com/UAVSwarm/UAVSwarm-dataset)
- **D-FINE**: [Redefine Bounding Box Regression for Real-Time Object Detection](https://github.com/Peterande/D-FINE)
- **YOLOX**: [Exceeding YOLO Series in 2021](https://github.com/Megvii-BaseDetection/YOLOX)
- **Shrimp RL Tracker**: [SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)
