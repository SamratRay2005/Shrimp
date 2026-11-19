# Drone Swarm RL Trajectory Tracker (`RM_Shrimp_extension`): Complete First-Principles Study Guide & Engineering Reference

> **Purpose of this document**: A **single, self-contained reference** combining every piece of knowledge from `README.md`, `PLAN.md`, `DEVLOG.md`, and the codebase — explained from **first principles**. It starts from the physics and mathematics of multi-object tracking, walks through every algorithmic pillar (Detection, Kalman Filtering, Dynamic Gating, Polar Subregions, Deep Q-Learning, and Bipartite Hungarian Optimization), documents the complete Phase 0–10 engineering history, cloud training logs, debugging breakthroughs, exact benchmarks, output schemas, and CLI commands so you only need to read `study.md`.

---

## Table of Contents

1. [The Core Problem: Why Tracking Drone Swarms Is Hard](#1-the-core-problem-why-tracking-drone-swarms-is-hard)
2. [From `Shrimp` to `RM_Shrimp_extension`: Architectural Evolution](#2-from-shrimp-to-rm_shrimp_extension-architectural-evolution)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Pillar 1 — Data Engineering & Leakage-Free Splits](#4-pillar-1--data-engineering--leakage-free-splits)
5. [Pillar 2 — Object Detection: D-FINE Transformer vs. YOLOX-S CNN](#5-pillar-2--object-detection-d-fine-transformer-vs-yolox-s-cnn)
6. [Pillar 3 — State Estimation: The 2D Constant-Velocity Kalman Filter](#6-pillar-3--state-estimation-the-2d-constant-velocity-kalman-filter)
7. [Pillar 4 — Velocity-Adaptive Dynamic Gating & 8-Subregion Geometry](#7-pillar-4--velocity-adaptive-dynamic-gating--8-subregion-geometry)
8. [Pillar 5 — Reinforcement Learning (DQN) for Data Association](#8-pillar-5--reinforcement-learning-dqn-for-data-association)
9. [Pillar 6 — Global Bipartite Optimization via the Hungarian Algorithm](#9-pillar-6--global-bipartite-optimization-via-the-hungarian-algorithm)
10. [Track Lifecycle Management & Trajectory Export](#10-track-lifecycle-management--trajectory-export)
11. [The Three Evaluation Pipelines (`run.py`, `run2.py`, `run3.py`)](#11-the-three-evaluation-pipelines-runpy-run2py-run3py)
12. [Empirical Benchmarks, Training Curves & Ablation Proof](#12-empirical-benchmarks-training-curves--ablation-proof)
13. [Complete Phase-by-Phase Engineering & Decision Log (Phases 0–10)](#13-complete-phase-by-phase-engineering--decision-log-phases-010)
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
| **Object Detector** | Single `last.onnx` shrimp model | **Dual Fine-Tuned Detectors**: D-FINE (`dfine_m`) & YOLOX-S (`ONNX`) | Tiny aerial drones ($<32\text{ px}$) need high small-object AP (`AP_small`) & fast inference. |
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
[Stage 1: Candidate Extraction]  ──► Pipeline 1 (run.py):  D-FINE HGNetv2-M Transformer
                                 ──► Pipeline 2 (run2.py): Shuffled GT Boxes (No IDs)
                                 ──► Pipeline 3 (run3.py): YOLOX-S (ONNX Runtime / det.txt)
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
   │  • Q-norm: q_norm = tanh((Q - 17.5) / 2.5) ∈ (-1, 1)
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

### 4.3 Bridge 2: MOT16 $\to$ Clean Trajectory CSV (`shrimp_drone/mot_to_csv.py`)

The RL tracker does **not** train on raw pixels — it trains on pure motion trajectories. `shrimp_drone/mot_to_csv.py` extracts `(frame, track_id, cx, cy, x1, y1, x2, y2, score)` across all sequences and filters out broken trajectories before RL training:

| Filter Rule | Threshold | Why It Matters from First Principles |
| :--- | :---: | :--- |
| **`min_length`** | $\ge 20$ frames | Ensures tracks are long enough to provide multi-step velocity context. |
| **`max_gap`** | $\le 2$ frames | Allows at most 1-frame temporary occlusion; rejects long dropouts that corrupt finite-difference velocity $(v_x, v_y)$. |
| **`max_jump`** | $\le 400.0\text{ px}$ | Rejects ground-truth ID glitches or teleportations between frames (up from `200px` in `Shrimp`). |

Running this across the 36 UAVSwarm training sequences converted **355 raw tracks / 49,661 detections** into **323 pristine continuous trajectories / 47,011 training points** (`drone_training_data.csv`).

---

## 5. Pillar 2 — Object Detection: D-FINE Transformer vs. YOLOX-S CNN

To avoid tying the RL tracker to a single detector paradigm, the project fine-tuned and benchmarked two fundamentally different object detector families on UAVSwarm.

### 5.1 Detector A: D-FINE (`HGNetv2-M` Hybrid Transformer) — Used in `run.py`

**D-FINE** (*Fine-grained Distribution Refinement for Real-Time Object Detection*) is a modern DETR-style (DEtection TRansformer) architecture:

1. **Backbone (`HGNetv2-B2`)**: Extracts multi-scale CNN feature maps at 3 pyramid levels (`return_idx: [1, 2, 3]`, channel widths `[384, 768, 1536]`) with `use_lab: True`.
2. **Hybrid Encoder**: Projects features to `hidden_dim: 256` (`depth_mult: 0.67`) and applies self-attention on the coarsest scale plus cross-scale feature fusion.
3. **D-FINE Transformer Decoder (`num_layers: 4`)**: Instead of regressing raw bounding-box coordinates $(x, y, w, h)$ as brittle point estimates, D-FINE predicts **probability distributions over bounding-box edge offsets** and iteratively refines those distributions across 4 decoder layers (Fine-grained Distribution Refinement, FDR).
4. **Deployment Constraint (`run.py`)**:
   - The hybrid encoder's positional embeddings (`pos_embed`) have a fixed size of **400 tokens**, corresponding to a $20 \times 20$ feature map at stride 32, i.e., $640 \times 640$ input resolution (`INPUT_W = 640`, `INPUT_H = 640`).
   - At runtime, `DFineDetector` converts BGR $\to$ RGB, resizes to `640x640`, runs `cfg.model.eval()` and `cfg.postprocessor.deploy()` (EMA weights from `checkpoints/dfine_drone_weights.pth`, `SCORE_THRES = 0.35`), and scales boxes back to `orig_size = (w, h)`.

### 5.2 Detector B: YOLOX-S (Anchor-Free Decoupled CNN + ONNX) — Used in `run3.py`

**YOLOX-S** (`depth=0.33, width=0.50`, 8.94M parameters) takes a completely different, purely convolutional approach:

1. **Anchor-Free Grid Prediction**:
   - At $640 \times 640$ input resolution, its Feature Pyramid Network outputs feature maps at 3 strides ($s \in \{8, 16, 32\}$), giving spatial grids of $80\times 80$, $40\times 40$, and $20\times 20$:
     $$80^2 + 40^2 + 20^2 = 6400 + 1600 + 400 = 8{,}400\text{ spatial cells}$$
   - Unlike older anchor-based YOLOs, each cell directly predicts one box without predefined anchor aspect ratios — advantageous for tiny variable-aspect drones.
2. **Decoupled Head**: Splits classification and bounding-box regression into separate convolutional branches so localization gradients do not interfere with objectness classification.
3. **ONNX Output Tensor & Post-Processing (`RuntimeYOLOXDetector.detect()` in `run3.py`)**:
   - The input image is **letterbox-resized** (aspect-ratio preserving resize + constant `114` gray padding to $640 \times 640$) without BGR$\to$RGB swap (matching YOLOX's BGR training convention).
   - The ONNX session produces predictions of shape `(8400, 4 + 1 + C)`:
     - Columns `0:4`: `[cx, cy, w, h]` in letterboxed $640 \times 640$ pixel space.
     - Column `4`: `obj_conf` (probability an object exists in this cell).
     - Columns `5:`: `cls_confs` (for single-class `yolox_drone_finetuned.onnx`, shape is `(8400, 6)`; for 80-class COCO baseline `yolox_s.onnx`, shape is `(8400, 85)`).
   - Final confidence per cell is `score = obj_conf * max(cls_confs)`.
   - Boxes above `conf_thresh` (default `0.05`) are divided by `scale` to return to original video coordinates, then deduplicated using **Non-Maximum Suppression (`cv2.dnn.NMSBoxes`, `nms_thresh=0.45`)**.

> **Why YOLOX-S outperformed D-FINE on UAVSwarm**: Trained for 12 epochs (`Mosaic=1.0`, `MixUp=0.5`, `Scale=(0.5, 1.5)`, Cosine LR, FP16), YOLOX's high-resolution stride-8 FPN head ($80\times 80$ grid) preserved tiny drone edges better than D-FINE's transformer tokens, boosting $\text{AP}_{\text{small}}$ from $46.0\%$ to **$52.2\%$** and $\text{AP}_{75}$ from $42.8\%$ to **$52.9\%$**, at **$>2\times$ faster speed** ($12.3\text{ ms}$ vs. $25.5\text{ ms}$) and **$9\times$ smaller model size** ($35.85\text{ MB}$ ONNX vs. $313.5\text{ MB}$ full checkpoint / $75\text{ MB}$ EMA weights). Meanwhile, D-FINE kept slightly higher raw recall ($96.0\%$ vs. $93.8\%$).

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
 (Fine-Tuned D-FINE)           (Ground Truth Oracle)           (YOLOX-S Runtime/Bench)
         │                               │                               │
   HGNetv2-M PyTorch           GT Boxes w/ IDs Stripped        ONNX Runtime (.onnx) OR
   (640x640, conf>=0.35)        & Shuffled Every Frame         Official det.txt (conf>=0.05)
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
   - **Purpose**: Evaluates the faster, lighter **YOLOX-S** detector in two modes:
     - `--mode runtime`: Live ONNX Runtime inference using our fine-tuned `checkpoints/yolox_drone_finetuned.onnx` (falls back to COCO `checkpoints/yolox_s.onnx` if absent).
     - `--mode benchmark`: Loads the official precomputed `UAVSwarm` benchmark YOLOX detections (`results/uavswarm_*_yolox_det.txt`) for apples-to-apples literature comparison.

| Metric / Property | Pipeline 1 (`run.py`: D-FINE + RL) | Pipeline 2 (`run2.py`: GT + RL) | Pipeline 3 (`run3.py`: YOLOX-S + RL) |
| :--- | :---: | :---: | :---: |
| **Detector** | D-FINE HGNetv2-M Transformer | Oracle Bounding Boxes (Shuffled, No IDs) | Fine-Tuned YOLOX-S CNN |
| **Parameters** | $19.19\text{ M}$ ($19{,}187{,}359$) | N/A (Ground Truth) | **$8.94\text{ M}$** |
| **Model Size** | $313.5\text{ MB}$ full ckpt ($75\text{ MB}$ EMA `.pth`) | N/A | **$35.85\text{ MB}$ (`.onnx`)** |
| **GPU Latency (Tesla T4)** | $25.5\text{ ms}$ ($\approx 39\text{ FPS}$) | $< 1\text{ ms}$ (direct memory) | **$12.3\text{ ms}$ ($\approx 81\text{ FPS}$)** |
| **$\text{mAP}@[0.50:0.95]$** | $48.2\%$ | $100.0\%$ (Upper bound) | **$53.44\%$** |
| **$\text{AP}_{50}$** | $92.6\%$ | $100.0\%$ | **$94.0\%$** |
| **$\text{AP}_{75}$** | $42.8\%$ | $100.0\%$ | **$52.9\%$** |
| **$\text{AP}_{\text{small}}$ ($<32\text{ px}$)** | $46.0\%$ | $100.0\%$ | **$52.2\%$** |
| **Primary Output Folder** | `result_with_our_dfine/` | `result_without_custom_object_detection/` | `result_with_yolox/` |
| **Ideal Use Case** | Robust global context, high recall | Empirical upper-bound tracking evaluation | High-speed real-time edge drone deployment |

---

## 12. Empirical Benchmarks, Training Curves & Ablation Proof

### 12.1 Detector Comparison (UAVSwarm Validation Split: 3,394 images, 36 sequences, 24,265 annotations)

| Performance Dimension | Fine-Tuned D-FINE (HGNetv2-M) | Fine-Tuned YOLOX-S | Difference / Winner |
| :--- | :---: | :---: | :--- |
| **Detector Architecture** | Hybrid Transformer (CNN + Attention + FDR) | Anchor-Free Decoupled CNN | Different trade-offs |
| **Model Parameters** | $19.19\text{ M}$ | **$8.94\text{ M}$** | **YOLOX is $53\%$ lighter** |
| **Checkpoint Size** | $313.5\text{ MB}$ full ($75\text{ MB}$ EMA weights) | **$35.85\text{ MB}$ (ONNX)** | **YOLOX is $9\times$ smaller** |
| **Inference Latency (T4 GPU)** | $25.5\text{ ms / frame}$ ($\sim 39\text{ FPS}$) | **$12.3\text{ ms / frame}$ ($\sim 81.3\text{ FPS}$)** | **YOLOX is $> 2\times$ faster** |
| **$\text{AP}_{50}$ (Drone Detection)** | $92.6\%$ | **$94.0\%$** | **YOLOX ($+1.4\%$)** |
| **$\text{AP}_{75}$ (Strict Localization)** | $42.8\%$ | **$52.9\%$** | **YOLOX ($+10.1\%$)** |
| **$\text{AP}_{\text{small}}$ (Drones $< 32\text{ px}$)** | $46.0\%$ | **$52.2\%$** | **YOLOX ($+6.2\%$)** |
| **$\text{AP}_{\text{medium}}$ ($32\text{--}96\text{ px}$)** | $63.6\%$ | **$64.7\%$** | **YOLOX ($+1.1\%$)** |
| **Overall $\text{mAP}@[0.50:0.95]$** | $48.2\%$ | **$53.44\%$** | **YOLOX ($+5.24\%$)** |
| **Operational Recall ($\text{IoU}=0.50$)** | **$96.0\%$** | $93.8\%$ | **D-FINE ($+2.2\%$)** |
| **Precision** | **$92.0\%$** | $91.5\%$ | **D-FINE ($+0.5\%$)** |

*(Additional D-FINE `COCOeval_opt` metrics recorded in `DEVLOG.md`: Recall $= 0.914$, $\text{F1 Score} = 0.917$, $1{,}053$ state-dict tensors.)*

### 12.2 Training Convergence Logs (Colab Tesla T4 GPU)

1. **D-FINE HGNetv2-M Training (`PID 9677`, 4.5 GB VRAM, TensorBoard Logs)**:
   - Total loss dropped from **$39.11 \to 22.27$**.
   - Bounding box L1 loss (`loss_bbox`) dropped **$7\times$** from **$0.4775 \to 0.0687$**.
   - Generalized IoU loss (`loss_giou`) dropped from **$1.8226 \to 0.7566$**.
2. **YOLOX-S Cloud Fine-Tuning (`yolox-trainer`, Tesla T4 15.6 GB VRAM, 12 Epochs, FP16)**:

| Epoch | Total Loss | $\text{mAP}@[0.50:0.95]$ | $\text{AP}_{50}$ | $\text{AP}_{75}$ | $\text{AP}_{\text{small}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | $4.40$ | $32.1\%$ | $76.4\%$ | — | — |
| **Epoch 5** | $2.70$ | $51.4\%$ | $92.9\%$ | — | — |
| **Epoch 10** | $2.30$ | $53.1\%$ | $93.1\%$ | $52.0\%$ | — |
| **Epoch 12 (Final)** | **$2.00$** | **$53.44\%$** | **$94.0\%$** | **$52.9\%$** | **$52.2\%$** |

3. **RL Policy (`DroneAgent`) Training**:
   - Verified 17 simultaneous candidates per step (4 background random clutter + 12 real swarm drone hard negatives + 1 ground truth).
   - Benchmarked at 10 epochs in $3.63\text{ s}$ on Tesla T4; trained full **600 epochs** with $\epsilon: 1.0 \to 0.05$, saving `checkpoints/drone_tracker_policy_best.pth` (`39.8 KB`).

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

### 12.4 Multi-Sequence Benchmarks Across All Evaluation Runs

#### A. Headline Empirical Tracking Benchmarks (`README.md`)

| Sequence | Resolution | Frames | Actual Swarm Size | Mean Active Swarm (Peak) | Persistent Tracks | Coordinates Traced |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UAVSwarm-02** | $812 \times 428$ | $156$ | $20$ drones | $19.0$ / frame (Peak: $20$) | $20$ | **$2{,}983$ points** |
| **UAVSwarm-04** | $812 \times 428$ | $161$ | $22$ drones | $18.7$ / frame (Peak: $22$) | $22$ | **$3{,}062$ points** |
| **UAVSwarm-08** | $446 \times 270$ | $115$ | $12$ drones | $9.8$ / frame (Peak: $12$) | $17$ | **$1{,}148$ points** |
| **UAVSwarm-14** | $812 \times 428$ | $115$ | $20$ drones | $16.6$ / frame (Peak: $20$) | $22$ | **$1{,}936$ points** |

#### B. Per-Pipeline Measurements Logged in `DEVLOG.md`

| Pipeline / Mode | Sequence | FPS / Latency | Persistent Tracks | Points Traced | Mean Swarm (Peak) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 7 `run.py` (Precomputed D-FINE dets)** | `UAVSwarm-02` | $25.6\text{ FPS}$ | $71$ | $3{,}206$ | $20.1$ ($27$) |
| **Phase 7 `run.py` (Precomputed D-FINE dets)** | `UAVSwarm-04` | $33.1\text{ FPS}$ | $20$ | $3{,}034$ | $18.7$ ($19$) |
| **Phase 7 `run.py` (Precomputed D-FINE dets)** | `UAVSwarm-08` | $47.2\text{ FPS}$ | $11$ | $1{,}090$ | $9.4$ ($10$) |
| **Phase 7 `run.py` (Precomputed D-FINE dets)** | `UAVSwarm-14` | $28.1\text{ FPS}$ | $20$ | $1{,}886$ | $16.2$ ($20$) |
| **Phase 7 Totals (4 sequences)** | $547$ frames | **$33.5\text{ FPS}$ avg** | — | **$9{,}216$ total** | — |
| **Live Runtime `run.py` (Apple Silicon MPS)** | `UAVSwarm-08` | $1.6\text{ FPS}$ ($71.33\text{ s}$, $\sim 150\text{ ms/f}$) | $14$ | $1{,}109$ | $9.5$ ($11$) |
| **Pipeline 2 `run2.py` (Shuffled GT Oracle)** | `UAVSwarm-02` | $36.1\text{ FPS}$ | $20$ | $2{,}983$ | — |
| **Pipeline 2 `run2.py` (Shuffled GT Oracle)** | `UAVSwarm-04` | $37.1\text{ FPS}$ | $20$ | $3{,}034$ | — |
| **Pipeline 3 `run3.py --mode runtime` (YOLOX)** | `UAVSwarm-04` | Detected $14\text{--}22$ drones $>0.75$ conf | — | $3{,}062$ | $18.7$ ($22$) |
| **Pipeline 3 `run3.py --mode runtime` (YOLOX)** | `UAVSwarm-08` | Live ONNX Runtime | — | $1{,}148$ | $9.8$ ($12$) |
| **Pipeline 3 `run3.py --mode runtime` (YOLOX)** | `UAVSwarm-14` | Live ONNX Runtime | — | $1{,}936$ | $16.6$ ($20$) |

---

## 13. Complete Phase-by-Phase Engineering & Decision Log (Phases 0–10)

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
   - Downloaded `dfine_m_coco.pth` (`79.1 MB`) for transfer learning and trained D-FINE on Tesla T4 (`AP50 = 92.6%`, `AR50 = 96.0%`, `mAP = 48.2%`).
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
10. **Phase 9 — YOLOX-S Cloud Fine-Tuning & Dual-Mode `run3.py` (2026-11-03)**:
    - Provisioned Colab `yolox-trainer` (Tesla T4, 15.6 GB VRAM), cloned full `3.7 GB` `UAVSwarm-dataset` (`12,963` images across 72 sequences), fine-tuned single-class `YOLOX-S` on `train_half.json` (`3,450` train imgs / `25,396` anns) and `val_half.json` (`3,394` val imgs / `24,265` anns) for 12 epochs (`batch=16`, `FP16`, Cosine LR `0.01/64` per img, `min_lr_ratio=0.05`, `warmup_epochs=1`, `weight_decay=5e-4`, `momentum=0.9`, `degrees=10.0`, `translate=0.1`, `scale=(0.5, 1.5)`, `mosaic_prob=1.0`, `mixup_prob=0.5`, `hsv_prob=1.0`).
    - Achieved **`53.44%` mAP**, **`94.0%` $\text{AP}_{50}$**, **`52.9%` $\text{AP}_{75}$**, **`52.2%` $\text{AP}_{\text{small}}$**, **`12.30 ms/frame`** ($81.3\text{ FPS}$). Exported `checkpoints/yolox_drone_finetuned.onnx` (`35.85 MB`) and built `run3.py` (`--mode runtime` & `--mode benchmark`) outputting to `result_with_yolox/`.
11. **Phase 10 — Multi-Video Validation, Workspace Hygiene & Publication Docs (2026-11-03)**:
    - Validated `run3.py --mode runtime` across `04`, `08`, and `14`.
    - Purged temporary scratch scripts from `scripts/`, retained only clean benchmark inputs in `results/`, documented YOLOX training in `notebooks/04_yolox_finetune.py`, updated `README.md` / `PLAN.md` / `DEVLOG.md`, and committed weights via Git LFS (`drone-swarm-tracking`).

---

## 14. Critical Bugs, Cloud Hurdles & Engineering Solutions

A quick reference of every nontrivial bug encountered in `DEVLOG.md` and how each was solved:

| # | Symptom / Failure | Root Cause | Exact Fix Implemented |
| :---: | :--- | :--- | :--- |
| **1** | D-FINE cross-entropy target out-of-bounds crash | Raw UAVSwarm JSON used `category_id: 1` (`'pedestrian'`), while `num_classes: 1` + `remap_mscoco_category: False` expects `0`-indexed targets `[0]`. | Sanitized `train.json`, `val.json`, `full_train.json` to `category_id: 0`, name `'drone'`. |
| **2** | D-FINE YAML dataloader assertion error | Base `include/dataloader.yml` sets `total_batch_size`; specifying `batch_size` simultaneously triggers an exclusivity assertion. | Overrode `total_batch_size: 4` (instead of `batch_size`) in D-FINE config. |
| **3** | D-FINE runtime tensor shape error on arbitrary video sizes & offline stall | `HybridEncoder` uses a fixed `pos_embed` of `400` tokens ($20\times 20$ grid at stride 32 $\implies 640\times 640$). Default `pretrained: True` also tries downloading backbone weights. | Forced `T.Resize((640, 640))` in `DFineDetector` and set `pretrained: False` in `dfine_hgnetv2_m_uavswarm.yml`. |
| **4** | `colab download` socket drop on `313 MB` `best_stg1.pth` | Jupyter server contents API encodes files as base64 in a single JSON HTTP payload (`GET /api/contents/...`), stalling/closing Tornado buffers above ~`50 MB`. | Extracted EMA-only weights (`74.99 MB` `dfine_drone_weights.pth`), ran `split -b 15M` into `chunk_aa..ae`, downloaded chunks, reassembled locally via `cat`, and automated mirroring in `scripts/sync_backups.py`. |
| **5** | Colab long-running jobs timing out | `google-colab-cli` (`colab exec`) defaults to a `30.0 s` client timeout, and Colab VMs disconnect on idle. | Passed `--timeout 3600`, launched detached `nohup` jobs writing to `/content/*.log`, and mirrored continuously via `sync_backups.py --loop`. |
| **6** | Megvii YOLOX install failures on Colab Python 3.13 | `setup.py` asserted `torch` wasn't found inside isolated PEP-517 build env, and `onnx-simplifier` failed metadata build on Python 3.13. | Stripped `onnx-simplifier` from `requirements.txt` and installed with `pip install --no-build-isolation -e .`. |
| **7** | PyTorch Dynamo ONNX export broken for YOLOX | Modern PyTorch Dynamo-based ONNX export produced incomplete external references on YOLOX's head. | Used TorchScript tracing (`torch.onnx.export(..., dynamo=False, opset_version=14)`) + `weights_only=False` checkpoint load, producing self-contained `yolox_drone_finetuned.onnx` (`35.85 MB`, shape `[1, 8400, 6]`). |
| **8** | `DroneAgent.__init__()` unexpected keyword `epochs` | Training script passed `epochs` to `DroneAgent` constructor instead of only the training loop. | Removed `epochs` argument from `DroneAgent.__init__` in `shrimp_drone/train_drone_agent.py`. |

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
# Optional: --weights, --policy, --conf (default 0.35), --benchmark-gt

# ── Pipeline 2: Shuffled Ground-Truth Oracle + RL Tracker (result_without_custom_object_detection/) ──
python run2.py
python run2.py --video results/uavswarm_test_04_raw.mp4
python run2.py --video results/uavswarm_test_08_raw.mp4
python run2.py --video results/uavswarm_test_14_raw.mp4
python run2.py --video results/uavswarm_test_04_raw.mp4 --headless
# Optional: --gt path/to/gt.txt, --policy

# ── Pipeline 3A: Live YOLOX-S ONNX + RL Tracker (result_with_yolox/) ──
python run3.py --mode runtime
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_08_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_14_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode runtime --headless

# ── Pipeline 3B: Official UAVSwarm Benchmark YOLOX Detections + RL Tracker ──
python run3.py --mode benchmark
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode benchmark
# Optional: --weights, --det path/to/det.txt, --policy, --conf (default 0.05), --output, --csv, --json

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
├── run.py                           # Pipeline 1: Live D-FINE detection + RL tracking
├── run2.py                          # Pipeline 2: Ground truth detection (shuffled) + RL tracking
├── run3.py                          # Pipeline 3: Live YOLOX ONNX / benchmark det.txt + RL tracking
├── study.md                         # This complete first-principles guide & reference
├── PLAN.md                          # Phase 0-10 roadmap & architecture comparison
├── DEVLOG.md                        # Chronological engineering decision & debugging log
├── README.md                        # Project overview, benchmark tables, and CLI quickstart
│
├── checkpoints/                     # Trained detector & RL policy weights (Git LFS)
│   ├── dfine_drone_weights.pth      # Fine-tuned D-FINE EMA detector weights (~75 MB)
│   ├── yolox_drone_finetuned.onnx   # Fine-tuned YOLOX-S single-class ONNX (35.85 MB, 8.94M params)
│   ├── yolox_s.onnx                 # Pretrained COCO YOLOX-S baseline (35.86 MB)
│   ├── drone_tracker_policy_best.pth# Trained 5→128→64→1 RL association policy (39.8 KB)
│   └── test_drone_policy_best.pth   # Backup RL policy checkpoint (39.8 KB)
│
├── result_with_our_dfine/           # Outputs from Pipeline 1 (D-FINE + RL)
├── result_without_custom_object_detection/ # Outputs from Pipeline 2 (Shuffled GT + RL)
├── result_with_yolox/               # Outputs from Pipeline 3 (YOLOX + RL)
│
├── results/                         # Clean raw benchmark videos & official annotations
│   ├── uavswarm_test_raw.mp4        # UAVSwarm-02 raw video (156 frames, 812x428)
│   ├── uavswarm_test_04_raw.mp4     # UAVSwarm-04 raw video (161 frames, 812x428)
│   ├── uavswarm_test_08_raw.mp4     # UAVSwarm-08 raw video (115 frames, 446x270)
│   ├── uavswarm_test_14_raw.mp4     # UAVSwarm-14 raw video (115 frames, 812x428)
│   ├── uavswarm_*_gt.txt            # Official MOT16 ground-truth files (02, 04, 08, 14)
│   └── uavswarm_*_yolox_det.txt     # Official benchmark YOLOX detections (02, 04, 08, 14)
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
│   └── 04_yolox_finetune.py         # YOLOX-S fine-tuning & ONNX export
│
├── logs/                            # Training terminal logs & summaries
└── scripts/
    └── sync_backups.py              # One-shot / --loop Colab backup mirror script
```

### 16.2 Core Stack & Citations

- **Stack**: Python 3.10+, PyTorch 2.0+, ONNX Runtime, OpenCV, NumPy, Pandas, SciPy (`linear_sum_assignment`), Pillow, PyYAML, Loguru.
- **UAVSwarm Dataset**: [A Benchmark for Multi-UAV Tracking in Complex Aerial Scenarios](https://github.com/UAVSwarm/UAVSwarm-dataset)
- **D-FINE**: [Redefine Bounding Box Regression for Real-Time Object Detection](https://github.com/Peterande/D-FINE)
- **YOLOX**: [Exceeding YOLO Series in 2021](https://github.com/Megvii-BaseDetection/YOLOX)
- **Shrimp RL Tracker**: [SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)
