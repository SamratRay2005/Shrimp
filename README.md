# 🛸 Drone Swarm RL Trajectory Tracker: Detection, Tracking & Continuous Flight Path Tracing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-brightgreen.svg)](https://onnxruntime.ai/)
[![D-FINE](https://img.shields.io/badge/Model-D--FINE--HGNetv2--M-green.svg)](https://github.com/Peterande/D-FINE)
[![YOLOX](https://img.shields.io/badge/Model-YOLOX--S--Finetuned-blue.svg)](https://github.com/Megvii-BaseDetection/YOLOX)
[![Benchmark](https://img.shields.io/badge/Dataset-UAVSwarm-orange.svg)](https://github.com/UAVSwarm/UAVSwarm-dataset)

An autonomous multi-target drone tracking and spatial trajectory tracing system. Adapted from the **Shrimp RL Tracker** architecture ([SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)) and substantially redesigned for **high-density aerial drone swarms** under the [UAVSwarm benchmark](https://github.com/UAVSwarm/UAVSwarm-dataset).

---

## 🌟 Overview & Key Innovations

In dense multi-UAV swarms, drones fly in close proximity ($60\text{--}90\text{ px}$ inter-drone separation), share nearly identical visual profiles, and perform abrupt velocity shifts. Traditional tracking methods relying on fixed Euclidean gating or greedy association suffer from frequent identity swaps and trajectory discontinuities.

This system addresses these challenges through a coupled detection, dynamic physical gating, and reinforcement learning framework:

1. **Dual Fine-Tuned Object Detectors (D-FINE & YOLOX)**:
   * **D-FINE HGNetv2-M Transformer**: $\text{AP}_{50} = 92.62\%$, providing robust global context and high-confidence box filtering.
   * **Fine-Tuned YOLOX-S**: $\text{mAP} = \mathbf{53.44\%}$, $\text{AP}_{50} = \mathbf{94.0\%}$, $\text{AP}_{\text{small}} = \mathbf{52.2\%}$, running at **$81+$ FPS** on GPU ($12.3\text{ ms}$ latency) with tightly fit bounding boxes ($\text{AP}_{75} = 52.9\%$).
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
  (Fine-Tuned D-FINE)           (Ground Truth Detection)           (Fine-Tuned YOLOX)
             │                               │                               │
   HGNetv2-M Transformer          Anonymous GT BBoxes Only        YOLOX-S ONNX (Runtime/Bench)
  AP50: 92.6% | Latency: 25ms    Per-frame shuffled boxes         AP50: 94.0% | Latency: 12ms
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

Evaluated on the exact same official UAVSwarm benchmark validation split (3,394 images across 36 test sequences, 24,265 drone ground-truth annotations):

| Performance Dimension | Our Fine-Tuned D-FINE (HGNetv2-M) | Our Fine-Tuned YOLOX (YOLOX-S) | Difference / Winner |
| :--- | :---: | :---: | :---: |
| **Detector Architecture** | Hybrid Transformer (CNN + Attention) | Anchor-Free Decoupled CNN | Different trade-offs |
| **Model Parameters** | 19.19 M | **8.94 M** | **YOLOX is 53% lighter** |
| **Checkpoint Size** | 313.5 MB | **35.85 MB (ONNX)** | **YOLOX is 9× smaller** |
| **Inference Latency (T4 GPU)**| 25.5 ms / frame ($\sim 39$ FPS) | **12.3 ms / frame ($\sim 81$ FPS)** | **YOLOX is > 2× faster** |
| **$\text{AP}_{50}$ (Drone Detection)**| 92.6% | **94.0%** | **YOLOX (+1.4%)** |
| **$\text{AP}_{75}$ (Strict Localization)**| 42.8% | **52.9%** | **YOLOX (+10.1%)** |
| **$\text{AP}_{\text{small}}$ (Drones $< 32\text{px}$)**| 46.0% | **52.2%** | **YOLOX (+6.2%)** |
| **$\text{AP}_{\text{medium}}$ ($32\text{--}96\text{px}$)**| 63.6% | **64.7%** | **YOLOX (+1.1%)** |
| **Overall mAP @ [0.50:0.95]**| 48.2% | **53.44%** | **YOLOX (+5.24%)** |
| **Operational Recall ($\text{IoU}=0.50$)**| **96.0%** | 93.8% | **D-FINE (+2.2%)** |
| **Precision** | **92.0%** | 91.5% | **D-FINE (+0.5%)** |

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
To prevent greedy track collisions, candidate assignments are framed as a global bipartite graph matching problem:
$$C_{i, j} = \|\mathbf{p}_i - \mathbf{z}_j\|_2 \times \Big( 1.5 - Q(\mathbf{s}_{i, j}) \Big)$$
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
│   ├── dfine_drone_weights.pth      # Fine-tuned D-FINE detector weights (313 MB)
│   ├── yolox_drone_finetuned.onnx   # Fine-tuned YOLOX-S ONNX model (35.8 MB, 8.94M params)
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
└── notebooks/                       # Colab GPU fine-tuning pipelines
    ├── 01_dataset_preparation.py    # UAVSwarm COCO conversion
    ├── 02_dfine_finetune.py         # D-FINE HGNetv2-M fine-tuning
    ├── 03_rl_agent_training.py      # RL Q-Network agent training
    └── 04_yolox_finetune.py         # YOLOX-S fine-tuning & ONNX export
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

Evaluated on official test sequences using our RL Tracker:

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
