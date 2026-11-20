# 🛸 Drone Swarm RL Trajectory Tracker: Detection, Tracking & Continuous Flight Path Tracing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-brightgreen.svg)](https://onnxruntime.ai/)
[![D-FINE](https://img.shields.io/badge/Model-D--FINE--HGNetv2--M-green.svg)](https://github.com/Peterande/D-FINE)
[![YOLOX](https://img.shields.io/badge/Model-YOLOX--M--Finetuned-blue.svg)](https://github.com/Megvii-BaseDetection/YOLOX)
[![Benchmark](https://img.shields.io/badge/Dataset-UAVSwarm-orange.svg)](https://github.com/UAVSwarm/UAVSwarm-dataset)

An autonomous multi-target drone tracking and spatial trajectory tracing system. Adapted from the **Shrimp RL Tracker** architecture ([SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)) and substantially redesigned for **high-density aerial drone swarms** under the [UAVSwarm benchmark](https://github.com/UAVSwarm/UAVSwarm-dataset).

---

## 🌟 Overview & Key Innovations

In dense multi-UAV swarms, drones fly in close proximity ($60\text{--}90\text{ px}$ inter-drone separation), share nearly identical visual profiles, and perform abrupt velocity shifts. Traditional tracking methods relying on fixed Euclidean gating or greedy association suffer from frequent identity swaps and trajectory discontinuities.

This system addresses these challenges through a coupled detection, dynamic physical gating, and reinforcement learning framework:

1. **Dual Cloud Fine-Tuned Object Detectors (D-FINE-M & YOLOX-M, `640×640`)**:
   * **Fine-Tuned D-FINE HGNetv2-M**: Unseen official 5,754-frame test $\text{AP}_{50} = \mathbf{92.86\%}$, $\text{mAP} = \mathbf{48.94\%}$, $\text{AP}_{75} = \mathbf{46.14\%}$, $\text{AP}_{\text{small}} = \mathbf{46.47\%}$ ($28.8\text{ FPS}$ on NVIDIA L4, `conf=0.55`).
   * **Fine-Tuned YOLOX-M**: Unseen official 5,754-frame test $\text{AP}_{50} = 90.28\%$, $\text{mAP} = 44.84\%$, $\text{AP}_{\text{small}} = 42.07\%$, running **$2\times$ faster at $57.0\text{ FPS}$** on NVIDIA L4 ($17.5\text{ ms}$ latency, `conf=0.50`).
2. **Velocity-Adaptive Dynamic Gating**: A kinematic search envelope $R_{\text{dynamic}} \in [20, 65]\text{ px}$ that contracts to $20\text{ px}$ during hovering (physically sealing out adjacent swarm drones) and expands up to $65\text{ px}$ during high-speed accelerations.
3. **Deep Q-Network (RL) Data Association**: A learned 5-feature spatial-temporal policy (`DroneQNetwork`) trained with hard-negative swarm clutter to predict candidate matching scores.
4. **Global Optimal Bipartite Association**: Formulates swarm association as a minimum-weight bipartite matching problem solved globally via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
5. **Continuous Trajectory Tracing**: Real-time rendering of anti-aliased neon flight paths, dynamic gate rings, telemetry HUD, and complete CSV/JSON spatial trajectory logging across dedicated output evaluation directories.

---

## 🔬 Three Comparative Evaluation Pipelines

This repository provides three parallel evaluation pipelines to benchmark detection vs. tracking contributions:

```
                                  Input Drone Video Frame
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
      [ Pipeline 1 ]                  [ Pipeline 2 ]                  [ Pipeline 3 ]
         run.py                          run2.py                         run3.py
  (Fine-Tuned D-FINE-M)         (Ground Truth Detection)        (Fine-Tuned YOLOX-M)
             │                               │                               │
   HGNetv2-M (`conf=0.55`)        Anonymous GT BBoxes Only        YOLOX-M ONNX (`conf=0.50`)
  AP50: 92.9% | Latency: 34.8ms  Per-frame shuffled boxes         AP50: 90.3% | Latency: 17.5ms
             │                               │                               │
             └───────────────────────┬───────┴───────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Autonomous Drone RL Tracker    │
                    │  - Kinematic 2D Kalman Filter   │
                    │  - Velocity-Adaptive Dynamic Gate│
                    │  - 8-Subregion Spatial Features │
                    │  - Deep Q-Network Policy (RL)   │
                    │  - Hungarian Bipartite Matcher  │
                    └────────────────┬────────────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    result_with_our_dfine/    result_without_custom_     result_with_yolox/
                               object_detection/
```

1. **[run.py](run.py) (D-FINE + RL Tracker)**: Evaluates our fine-tuned transformer detector paired with our RL tracker. Outputs to `result_with_our_dfine/`.
2. **[run2.py](run2.py) (Ground Truth Detection + RL Tracker)**: Feeds isolated, ground-truth bounding boxes solely as single-frame detections (with track IDs removed and boxes shuffled per frame). Evaluates the pure upper-bound tracking capability of our RL tracker without detector noise. Outputs to `result_without_custom_object_detection/`.
3. **[run3.py](run3.py) (YOLOX + RL Tracker)**: Dual-mode pipeline supporting:
   * **Runtime Mode (`--mode runtime`)**: Live neural network forward passes via our newly fine-tuned [yolox_drone_finetuned.onnx](checkpoints/yolox_drone_finetuned.onnx).
   * **Benchmark Mode (`--mode benchmark`)**: Uses official precomputed UAVSwarm benchmark YOLOX detection files.
   * Outputs to `result_with_yolox/`.

---

## 📊 Comprehensive Detector Benchmark Comparison

Trained with strict zero test-set leakage (`train_85`: 5,805 train images; `val_15`: 1,039 images for checkpoint & threshold selection) and evaluated once on the **official unseen UAVSwarm test split (`5,754` images across `36` test sequences, `37,703` COCO annotations)** on NVIDIA L4 (24 GB):

| Performance Dimension | Fine-Tuned D-FINE (`HGNetv2-M`) | Fine-Tuned YOLOX (`YOLOX-M 640×640`) | Difference / Winner |
| :--- | :---: | :---: | :---: |
| **Detector Architecture** | Hybrid Transformer (CNN + Attention) | Anchor-Free Decoupled CNN | Different trade-offs |
| **Checkpoint / Artifact Size** | **78.57 MB (EMA `.pth`)** | 101.20 MB (ONNX) | **D-FINE is 22% smaller** |
| **Inference Latency (L4 GPU)** | 34.76 ms / frame (28.77 FPS) | **17.53 ms / frame (57.03 FPS)** | **YOLOX-M is 2.0× faster** |
| **$\text{AP}_{50}$ (Official Test)** | **92.86%** | 90.28% | **D-FINE (+2.58%)** |
| **$\text{AP}_{75}$ (Strict Localization)** | **46.14%** | 38.12% | **D-FINE (+8.02%)** |
| **$\text{AP}_{\text{small}}$ (Drones $< 32\text{px}$)** | **46.47%** | 42.07% | **D-FINE (+4.40%)** |
| **$\text{AP}_{\text{medium}}$ ($32\text{--}96\text{px}$)** | **60.84%** | 56.51% | **D-FINE (+4.33%)** |
| **Overall mAP @ `[0.50:0.95]`** | **48.94%** | 44.84% | **D-FINE (+4.10%)** |
| **$\text{AR}@100$ (Official Test)** | **59.94%** | 51.50% | **D-FINE (+8.44%)** |
| **Locked `val_15` Threshold / F1** | `conf = 0.55` (`F1 = 94.25%`) | `conf = 0.50` (**`F1 = 96.62%`**) | Both leak-free on `val_15` |

---

## 🧠 System Architecture & Mathematical Logic

### 1. Kinematic Prediction (2D Kalman Filter)
For every active drone $i$, state transitions follow a constant-velocity kinematic model:
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{P}_t = \mathbf{F} \mathbf{P}_{t-1} \mathbf{F}^T + \mathbf{Q}$$
$$\mathbf{F} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
where predicted coordinates are $(\hat{cx}_i, \hat{cy}_i) = (x_t[0], x_t[1])$.

### 2. Velocity-Adaptive Dynamic Gating
Static gating fails in swarms because inter-drone spacing ($60\text{--}90\text{ px}$) causes nearby drones to enter each other's association windows. We compute a dynamic envelope for each track:
$$R_{\text{dynamic}} = \mathrm{clip}\left( R_{\text{base}} + \alpha \cdot \|\mathbf{v}\|_2 + \beta \cdot k_{\text{missed}}, \;\; R_{\text{min}}, \;\; R_{\text{max}} \right)$$
* $R_{\text{base}} = 20.0\text{ px}$, $\alpha = 2.0$, $\beta = 4.0$
* $R_{\text{min}} = 20.0\text{ px}$ (locks out neighboring hovering drones)
* $R_{\text{max}} = 65.0\text{ px}$ (strictly lower than the minimum swarm inter-drone gap)

### 3. Spatial Discretization & RL Feature Representation
Candidates within $R_{\text{dynamic}}$ are mapped into 8 spatial subregions (2 concentric rings $\times$ 4 angular quadrants):
$$\text{ring} = \begin{cases} 0 & \text{if } d \le R_{\text{dynamic}} / 2 \\ 1 & \text{otherwise} \end{cases}, \quad \text{quadrant} = \left\lfloor \frac{\theta + \pi/4}{\pi/2} \right\rfloor \pmod 4$$
$$\text{region\_id} = 4 \cdot \text{ring} + \text{quadrant} \in \{0, 1, \dots, 7\}$$

The 5-dimensional feature input to `DroneQNetwork` is normalized as:
$$\mathbf{s} = \left[ \frac{\Delta x}{R_{\text{dynamic}}}, \; \frac{\Delta y}{R_{\text{dynamic}}}, \; \frac{\text{region\_id}}{7.0}, \; \frac{v_x}{R_{\text{dynamic}}}, \; \frac{v_y}{R_{\text{dynamic}}} \right]$$

### 4. Global Hungarian Cost Optimization
To prevent greedy track collisions, candidate assignments are framed as a global bipartite graph matching problem. Because the $\gamma = 0.95$ DQN outputs cumulative discounted returns $Q(\mathbf{s}_{i, j}) \in [16, 20]$, raw Q-values are first centered and bounded into $(-1, 1)$ before weighting Euclidean distance:
$$\tilde{Q}(\mathbf{s}_{i, j}) = \tanh\!\left(\frac{Q(\mathbf{s}_{i, j}) - 17.5}{2.5}\right), \qquad C_{i, j} = \|\mathbf{p}_i - \mathbf{z}_j\|_2 \times \Big( 1.5 - \tilde{Q}(\mathbf{s}_{i, j}) \Big)$$
The global optimal matching $(\mathbf{r}^*, \mathbf{c}^*)$ is computed via the Hungarian algorithm:
$$\min_{\mathbf{X}} \sum_{i} \sum_{j} C_{i, j} X_{i, j} \quad \text{s.t.} \quad \sum_j X_{i, j} \le 1, \; \sum_i X_{i, j} \le 1, \; X_{i, j} \in \{0, 1\}$$

---

## 📁 Repository Structure

```
RM_Shrimp_extension/
│
├── run.py                           # Pipeline 1: Live D-FINE detection + RL tracking
├── run2.py                          # Pipeline 2: Ground truth detection + RL tracking
├── run3.py                          # Pipeline 3: Live YOLOX detection / benchmark + RL tracking
│
├── checkpoints/                     # Model weights & policies
│   ├── dfine_drone_weights.pth      # Fine-tuned D-FINE-M EMA weights (78.57 MB)
│   ├── yolox_drone_finetuned.onnx   # Fine-tuned YOLOX-M ONNX model (101.20 MB)
│   ├── yolox_s.onnx                 # Pretrained COCO baseline (34.5 MB)
│   └── drone_tracker_policy_best.pth# Trained RL association policy (39.8 KB)
│
├── result_with_our_dfine/           # Evaluation outputs from Pipeline 1 (D-FINE)
│   ├── tracked_*.mp4
│   └── trajectories_*.csv / .json
│
├── result_without_custom_object_    # Evaluation outputs from Pipeline 2 (Ground Truth)
│   detection/
│   ├── tracked_*.mp4
│   └── trajectories_*.csv / .json
│
├── result_with_yolox/               # Evaluation outputs from Pipeline 3 (YOLOX)
│   ├── tracked_*.mp4
│   └── trajectories_*.csv / .json
│
├── results/                         # Raw benchmark videos & annotations (Clean inputs)
│   ├── uavswarm_test_raw.mp4        # UAVSwarm-02 test video (156 frames)
│   ├── uavswarm_test_04_raw.mp4     # UAVSwarm-04 test video (161 frames)
│   ├── uavswarm_test_08_raw.mp4     # UAVSwarm-08 test video (115 frames)
│   ├── uavswarm_test_14_raw.mp4     # UAVSwarm-14 test video (115 frames)
│   ├── uavswarm_*_gt.txt            # Ground-truth annotations
│   └── uavswarm_*_yolox_det.txt     # Benchmark YOLOX detection outputs
│
├── shrimp_drone/                    # Core RL tracking & environment modules
│   ├── drone_env.py                 # Gym environment with dynamic gating & hard negatives
│   ├── train_drone_agent.py         # DQN agent training script
│   └── mot_to_csv.py                # Dataset parser & MOT-to-CSV converter
│
├── dataset/                         # Dataset preparation utilities
│   ├── download_uavswarm.py         # Automated UAVSwarm downloader
│   └── mot2coco.py                  # MOT16 to COCO format converter
│
├── dfine_core/                      # D-FINE model architecture & configs
│
├── evaluations/                     # Organized 36-sequence benchmarks & strategy ablations
│   ├── mot_metrics.py               # Unit-tested CLEAR MOT, IDF1 & HOTA evaluators
│   ├── evaluate_tracking.py         # Tier-A oracle-box benchmark & ablations
│   ├── evaluate_end_to_end.py       # Tier-B D-FINE-M / YOLOX-M end-to-end benchmark
│   ├── results.md                   # Full Tier-A/D/B & cost-strategy report
│   ├── data/                        # uavswarm_test_gt/, detections_dfine/, detections_yolox/
│   ├── results/                     # tier_a/, tier_b/, predictions/, plots/
│   └── testing_different_stratergies/
│       ├── evaluate_neg_q.py
│       ├── evaluate_neg_q_over_d.py
│       ├── evaluate_d_over_q.py
│       └── results/                 # Strategy CSV & JSON comparisons
│
├── comparision.md                   # 14-paper (2022–2026) UAVSwarm literature benchmark
│
└── notebooks/                       # Colab GPU fine-tuning pipelines
    ├── 01_dataset_preparation.py    # UAVSwarm COCO conversion
    ├── 02_dfine_finetune.py         # D-FINE HGNetv2-M fine-tuning
    ├── 03_rl_agent_training.py      # RL Q-Network agent training
    └── 04_yolox_finetune.py         # YOLOX-M fine-tuning & ONNX export
```

---

## 🚀 Quickstart & Usage

### 1. Installation & Environment Setup
```bash
git clone https://github.com/SamratRay2005/Shrimp.git
cd Shrimp
pip install torch torchvision opencv-python onnxruntime numpy pandas scipy pillow pyyaml loguru
```

### 2. Running Pipeline 1: D-FINE Detector + RL Tracker
```bash
# Run on default test video with interactive GUI
python run.py

# Run on specific test sequence
python run.py --video results/uavswarm_test_04_raw.mp4
python run.py --video results/uavswarm_test_08_raw.mp4
python run.py --video results/uavswarm_test_14_raw.mp4

# Run headless (directly writes MP4, CSV, and JSON to result_with_our_dfine/)
python run.py --video results/uavswarm_test_04_raw.mp4 --headless
```

### 3. Running Pipeline 2: Ground Truth Detection + RL Tracker
```bash
# Run on default test video
python run2.py

# Run on specific benchmark sequence (automatically matches gt.txt)
python run2.py --video results/uavswarm_test_04_raw.mp4
python run2.py --video results/uavswarm_test_08_raw.mp4
python run2.py --video results/uavswarm_test_14_raw.mp4

# Run headless (directly writes MP4, CSV, and JSON to result_without_custom_object_detection/)
python run2.py --video results/uavswarm_test_04_raw.mp4 --headless
```

### 4. Running Pipeline 3: YOLOX Detector + RL Tracker
```bash
# Mode A: Live runtime neural network inference using our fine-tuned ONNX model
python run3.py --mode runtime
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_08_raw.mp4 --mode runtime
python run3.py --video results/uavswarm_test_14_raw.mp4 --mode runtime

# Mode B: Official UAVSwarm benchmark YOLOX detection mode
python run3.py --mode benchmark
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode benchmark

# Run headless (directly writes MP4, CSV, and JSON to result_with_yolox/)
python run3.py --video results/uavswarm_test_04_raw.mp4 --mode runtime --headless
```

### 5. Interactive GUI Controls
When running with the live display active:
* **`SPACE`**: Pause and resume video playback.
* **`q`** or **`ESC`**: Terminate tracking cleanly and save current video/CSV logs.

---

## 📈 Empirical Tracking Benchmarks

### 36-Sequence Official UAVSwarm Benchmark Summary (`5,754` Frames, `325` GT Tracks, `37,947` Boxes)
Full per-sequence tables, ablation studies, and literature comparisons across 14 published papers (2022–2026) are documented in [evaluations/results.md](evaluations/results.md) and [comparision.md](comparision.md):

| Pipeline / Tier | Detector | MOTA (%) ↑ | HOTA (%) ↑ | IDF1 (%) ↑ | IDSW ↓ | Avg Purity (%) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier A (Oracle Tracking, `run2.py`)** | Shuffled GT boxes | **99.84** | **97.14** | **96.12** | 44 | **99.32** |
| **Tier B (End-to-End, `run.py`)** | **`D-FINE-M (640, conf=0.55)`** | **82.87** | **68.12** | 86.91 | 96 | **99.43** |
| **Tier B (End-to-End, `run3.py`)** | **`YOLOX-M (640, conf=0.50)`** | 81.94 | 66.85 | **87.18** | **54** | 99.42 |

### Sample Sequence Telemetry (`run2.py` / `results/`)
Evaluated on the 4 sample test sequences using our RL Tracker:

| Sequence | Resolution | Frames | Actual Swarm Size | Mean Active Swarm (Detected) | Persistent Tracks | Trajectory Coordinates Traced |
|---|---|---|---|---|---|---|
| **UAVSwarm-02** | $812 \times 428$ | 156 | 20 drones | 19.0 drones/frame (Peak: 20) | 20 | **2,983 points** |
| **UAVSwarm-04** | $812 \times 428$ | 161 | 22 drones | 18.7 drones/frame (Peak: 22) | 22 | **3,062 points** |
| **UAVSwarm-08** | $446 \times 270$ | 115 | 12 drones | 9.8 drones/frame (Peak: 12) | 17 | **1,148 points** |
| **UAVSwarm-14** | $812 \times 428$ | 115 | 20 drones | 16.6 drones/frame (Peak: 20) | 22 | **1,936 points** |

---

## 📄 Output Data Formats

Each pipeline generates standardized CSV and JSON trajectory datasets in its dedicated directory:

### CSV Output Format (`trajectories_<video>.csv`)
```csv
track_id,frame,cx,cy,vx,vy,x1,y1,x2,y2
1,1,278.76,124.65,0.0,0.0,260.87,110.35,296.65,138.96
1,2,278.28,125.90,-0.48,1.25,260.09,110.96,296.47,140.84
...
```

### JSON Output Format (`trajectories_<video>.json`)
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

## 🔬 Citations & References

If you use this work or the underlying algorithms in your research, please reference:
* **UAVSwarm Dataset**: [A Benchmark for Multi-UAV Tracking in Complex Aerial Scenarios](https://github.com/UAVSwarm/UAVSwarm-dataset)
* **D-FINE**: [Redefine Bounding Box Regression for Real-Time Object Detection](https://github.com/Peterande/D-FINE)
* **YOLOX**: [Exceeding YOLO Series in 2021](https://github.com/Megvii-BaseDetection/YOLOX)
* **Shrimp RL Tracker**: [SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)
