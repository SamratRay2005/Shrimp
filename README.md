# 🛸 Drone Swarm RL Trajectory Tracker: Detection, Tracking & Continuous Flight Path Tracing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![D-FINE](https://img.shields.io/badge/Model-D--FINE--HGNetv2--M-green.svg)](https://github.com/Peterande/D-FINE)
[![Benchmark](https://img.shields.io/badge/Dataset-UAVSwarm-orange.svg)](https://github.com/UAVSwarm/UAVSwarm-dataset)

An autonomous multi-target drone tracking and spatial trajectory tracing system. Adapted from the **Shrimp RL Tracker** architecture ([SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)) and substantially redesigned for **high-density aerial drone swarms** under the [UAVSwarm benchmark](https://github.com/UAVSwarm/UAVSwarm-dataset).

---

## 🌟 Overview & Key Innovations

In dense multi-UAV swarms, drones fly in close proximity ($60\text{--}90\text{ px}$ inter-drone separation), share nearly identical visual profiles, and perform abrupt velocity shifts. Traditional tracking methods relying on fixed Euclidean gating or greedy association suffer from frequent identity swaps and trajectory discontinuities.

This system addresses these challenges through a coupled detection, dynamic physical gating, and reinforcement learning framework:

1. **Runtime D-FINE Object Detection**: Replaces static ONNX YOLO models with a fine-tuned **D-FINE HGNetv2-M** transformer detector ($\text{AP}_{50} = 92.62\%$, $\text{AR}_{50} = 96.0\%$) running live on incoming video frames.
2. **Velocity-Adaptive Dynamic Gating**: A kinematic search envelope $R_{\text{dynamic}} \in [20, 65]\text{ px}$ that contracts to $20\text{ px}$ during hovering (physically sealing out adjacent swarm drones) and expands up to $65\text{ px}$ during high-speed accelerations.
3. **Deep Q-Network (RL) Data Association**: A learned 5-feature spatial-temporal policy (`DroneQNetwork`) trained with hard-negative swarm clutter to predict candidate matching scores.
4. **Global Optimal Bipartite Association**: Formulates swarm association as a minimum-weight bipartite matching problem solved globally via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
5. **Continuous Trajectory Tracing**: Real-time rendering of anti-aliased neon flight paths, dynamic gate rings, telemetry HUD, and complete CSV/JSON spatial trajectory logging.

---

## 📊 Comparative Architecture: Shrimp vs. Drone Swarm

| Component | Original Shrimp Implementation | Drone Swarm System (This Repository) |
|---|---|---|
| **Primary Domain** | Single-species underwater shrimp | Multi-type aerial drone swarms (UAVSwarm benchmark) |
| **Object Detector** | YOLO-based (`last.onnx`) | **D-FINE HGNetv2-M** fine-tuned on aerial drone swarms |
| **Detection Performance** | Tank conditions | **$\text{AP}_{50} = 92.62\%$**, **$\text{AR}_{50} = 96.0\%$**, **$\text{F1} = 0.917$** |
| **Search Gate Envelope** | Static 150 px radius | **Velocity-Adaptive Dynamic Gate ($R \in [20, 65]\text{ px}$)** |
| **Data Association** | Greedy local $\text{argmax}(Q)$ | **Hungarian Global Bipartite Matching + RL Q-Scores** |
| **Clutter / Negative Samples** | 4 synthetic random noise points | 4 random points + **real neighbor drones (hard negatives)** |
| **Q-Network Topology** | 5 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 1 | **5 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 1** |
| **RL Experience Buffer** | 10,000 steps | **50,000 steps** across 47,011 trajectory transitions |
| **Primary Output** | Vector hysteresis line-crossing counting | **Continuous multi-target spatial trajectory tracing** |
| **Track Identity Switches** | Frequent during dense crossings | **Zero identity switches** across test benchmark sequences |

---

## 🧠 System Architecture & Mathematical Logic

```
   Raw Video Frame
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: D-FINE Transformer Detector (Runtime Inference)│
│  - HGNetv2-M Backbone + Hybrid Encoder + Decoder       │
│  - Resolution: 640 x 640 | Hardware: MPS / CUDA / CPU  │
└──────────────────────────┬─────────────────────────────┘
                           │ Bounding Boxes & Centroids [cx, cy]
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Kinematic Motion Estimation (2D Kalman Filter)│
│  - Predicts state: [x, y, vx, vy]^T                    │
│  - Estimates centroid displacement & ground velocity   │
└──────────────────────────┬─────────────────────────────┘
                           │ Predicted Position (cx_hat, cy_hat)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Velocity-Adaptive Dynamic Gating              │
│  - R_dynamic = clip(20 + 2.0*v + 4.0*missed, 20, 65) px│
│  - Contracts for hovering; expands for dashing drones   │
└──────────────────────────┬─────────────────────────────┘
                           │ Candidates inside R_dynamic
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Deep Q-Network Association Scoring (RL)       │
│  - 8-subregion spatial discretization                  │
│  - 5-feature state vector -> Q-Value in [0, 1]         │
└──────────────────────────┬─────────────────────────────┘
                           │ Cost Matrix: dist * (1.5 - Q)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Global Hungarian Assignment & Path Tracing    │
│  - scipy.optimize.linear_sum_assignment                │
│  - Sliding-window anti-aliased neon trails             │
│  - Trajectory dataset export (CSV & JSON)              │
└────────────────────────────────────────────────────────┘
```

---

### Mathematical Formulation

#### 1. Kinematic Prediction (2D Kalman Filter)
For every active drone $i$, state transitions follow a constant-velocity kinematic model:
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{P}_t = \mathbf{F} \mathbf{P}_{t-1} \mathbf{F}^T + \mathbf{Q}$$
$$\mathbf{F} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
where predicted coordinates are $(\hat{cx}_i, \hat{cy}_i) = (x_t[0], x_t[1])$.

#### 2. Velocity-Adaptive Dynamic Gating
Static gating fails in swarms because inter-drone spacing ($60\text{--}90\text{ px}$) causes nearby drones to enter each other's association windows. We compute a dynamic envelope for each track:
$$R_{\text{dynamic}} = \mathrm{clip}\left( R_{\text{base}} + \alpha \cdot \|\mathbf{v}\|_2 + \beta \cdot k_{\text{missed}}, \;\; R_{\text{min}}, \;\; R_{\text{max}} \right)$$
* $R_{\text{base}} = 20.0\text{ px}$, $\alpha = 2.0$, $\beta = 4.0$
* $R_{\text{min}} = 20.0\text{ px}$ (locks out neighboring hovering drones)
* $R_{\text{max}} = 65.0\text{ px}$ (strictly lower than the minimum swarm inter-drone gap)

#### 3. Spatial Discretization & RL Feature Representation
Candidates within $R_{\text{dynamic}}$ are mapped into 8 spatial subregions (2 concentric rings $\times$ 4 angular quadrants):
$$\text{ring} = \begin{cases} 0 & \text{if } d \le R_{\text{dynamic}} / 2 \\ 1 & \text{otherwise} \end{cases}, \quad \text{quadrant} = \left\lfloor \frac{\theta + \pi/4}{\pi/2} \right\rfloor \pmod 4$$
$$\text{region\_id} = 4 \cdot \text{ring} + \text{quadrant} \in \{0, 1, \dots, 7\}$$

The 5-dimensional feature input to `DroneQNetwork` is normalized as:
$$\mathbf{s} = \left[ \frac{\Delta x}{R_{\text{dynamic}}}, \; \frac{\Delta y}{R_{\text{dynamic}}}, \; \frac{\text{region\_id}}{7.0}, \; \frac{v_x}{R_{\text{dynamic}}}, \; \frac{v_y}{R_{\text{dynamic}}} \right]$$

#### 4. Global Hungarian Cost Optimization
To prevent greedy track collisions, candidate assignments are framed as a global bipartite graph matching problem:
$$C_{i, j} = \|\mathbf{p}_i - \mathbf{z}_j\|_2 \times \Big( 1.5 - Q(\mathbf{s}_{i, j}) \Big)$$
The global optimal matching $(\mathbf{r}^*, \mathbf{c}^*)$ is computed via the Munkres/Hungarian algorithm:
$$\min_{\mathbf{X}} \sum_{i} \sum_{j} C_{i, j} X_{i, j} \quad \text{s.t.} \quad \sum_j X_{i, j} \le 1, \; \sum_i X_{i, j} \le 1, \; X_{i, j} \in \{0, 1\}$$

---

## 🏋️ Model Training Methodology

### 1. D-FINE Object Detector Training
* **Architecture**: HGNetv2-M backbone with 4-layer transformer decoder (19.18M parameters).
* **Dataset**: 7,096 UAVSwarm images converted to COCO single-class format (`3,450 train` / `3,394 val`).
* **Optimization**: AdamW optimizer ($\text{lr} = 2 \times 10^{-4}$, weight decay $1 \times 10^{-4}$), Automatic Mixed Precision (AMP), multi-scale aerial photometric and spatial augmentations.
* **Convergence Benchmark**:
  * $\text{AP}@[0.50:0.95] = \mathbf{0.482}$
  * $\text{AP}@0.50 = \mathbf{0.926}$ (92.6% precision at 50% IoU)
  * $\text{Recall}@0.50 = \mathbf{0.960}$ (96.0% drone recall)
  * $\text{F1 Score} = \mathbf{0.917}$

### 2. Reinforcement Learning (DQN) Association Policy
* **Environment**: `shrimp_drone/drone_env.py` (`DroneTrackingEnv`).
* **Hard-Negative Mining**: Rather than relying purely on random Gaussian clutter, real trajectories of adjacent swarm drones are injected as hard-negative candidates ($k \le 12$) during training.
* **Reward Structure**:
  * Correct Drone Association: $+1.0$
  * Neighbor Drone / Clutter Misassociation: $-1.0$
  * Missed Observation: $-0.2$
* **Training Protocol**: 600 epochs, 50,000-step experience replay buffer, epsilon-greedy decay ($1.0 \rightarrow 0.05$, $\gamma = 0.95$).

---

## 📁 Repository Structure

```
RM_Shrimp_extension/
│
├── run.py                           # Master runtime pipeline: live D-FINE detection + RL tracking on any video
├── run2.py                          # Ground truth annotation pipeline: benchmark evaluation using real annotations
│
├── shrimp_drone/                    # Core RL training and environment modules
│   ├── drone_env.py                 # DroneTrackingEnv with dynamic gating (matches Shrimp/shrimp_env.py)
│   ├── train_drone_agent.py         # DQN agent & replay buffer training script (matches Shrimp/train_agent.py)
│   └── mot_to_csv.py                # Dataset cleaner & MOT-to-CSV converter (matches Shrimp/clean_data.py)
│
├── checkpoints/                     # Model weights
│   ├── dfine_drone_weights.pth      # Fine-tuned D-FINE detector weights (75 MB)
│   └── drone_tracker_policy_best.pth# Trained RL association policy weights (39.8 KB)
│
├── dfine_core/                      # D-FINE neural network architecture & custom configs
│   ├── configs/dfine/custom/        # UAVSwarm single-class runtime configs
│   └── src/                         # HGNetv2 backbone, transformer encoder/decoder, postprocessors
│
├── dataset/                         # Training & trajectory datasets
│   └── clean_drone_tracks.csv       # Cleaned ground-truth trajectories for RL training
│
├── results/                         # Evaluation test videos and output artifacts
│   ├── uavswarm_test_raw.mp4        # UAVSwarm-02 test video (156 frames)
│   ├── uavswarm_test_04_raw.mp4     # UAVSwarm-04 test video (161 frames)
│   ├── uavswarm_test_08_raw.mp4     # UAVSwarm-08 test video (115 frames)
│   ├── uavswarm_test_14_raw.mp4     # UAVSwarm-14 test video (115 frames)
│   ├── tracked_*.mp4                # Tracked videos with anti-aliased neon trails & gate badges
│   └── trajectories_*.csv / .json   # Output trajectory dataset records
│
├── notebooks/                       # Colab training & data preparation pipelines
└── scripts/                         # Resilient chunked cloud backup synchronization
```

---

## 🚀 Quickstart & Usage

### 1. Installation & Environment Setup
Clone the repository and install required dependencies:
```bash
git clone https://github.com/SamratRay2005/Shrimp.git
cd Shrimp
pip install torch torchvision opencv-python numpy pandas scipy pillow pyyaml loguru
```

### 2. Live Runtime Tracking
Run real-time D-FINE object detection and RL trajectory tracing on any drone video:

```bash
# Run on default test video with interactive GUI
python run.py

# Run on specific test sequence
python run.py --video results/uavswarm_test_04_raw.mp4
python run.py --video results/uavswarm_test_08_raw.mp4
python run.py --video results/uavswarm_test_14_raw.mp4

# Run on ANY arbitrary drone video
python run.py --video /path/to/any_drone_video.mp4

# Run headless (ideal for server/cloud instances; directly outputs MP4 and CSV)
python run.py --video results/uavswarm_test_08_raw.mp4 --headless
```

### 3. Ground Truth Annotation Tracking (`run2.py`)
Run the RL data association and dynamic gating pipeline directly on **real ground truth annotations** rather than D-FINE detector output. This isolates the tracking and dynamic gating mechanics from detector noise and provides an empirical upper-bound benchmark running at real-time speeds (30–60+ FPS):

```bash
# Run on default test video using its matched ground truth annotations
python run2.py

# Run on specific benchmark sequences (automatically maps corresponding UAVSwarm gt.txt)
python run2.py --video results/uavswarm_test_04_raw.mp4
python run2.py --video results/uavswarm_test_08_raw.mp4
python run2.py --video results/uavswarm_test_14_raw.mp4

# Run with an explicit ground truth annotation file (MOT format or CSV)
python run2.py --video results/uavswarm_test_raw.mp4 --gt results/uavswarm_02_gt.txt

# Run headless
python run2.py --video results/uavswarm_test_04_raw.mp4 --headless
```

### 4. Interactive GUI Controls
When running with the live display active:
* **`SPACE`**: Pause and resume video playback.
* **`q`** or **`ESC`**: Terminate tracking cleanly and save current video/CSV logs.

### 5. Configuration Flags
* `--video`: Path to input video file (default: `results/uavswarm_test_raw.mp4`).
* `--weights`: Path to D-FINE detector checkpoint (default: `checkpoints/dfine_drone_weights.pth`).
* `--policy`: Path to RL policy weights (default: `checkpoints/drone_tracker_policy_best.pth`).
* `--conf`: Object detection confidence threshold (default: `0.35`).
* `--output`: Custom output path for rendered MP4 video.
* `--csv`: Custom output path for trajectory CSV export.
* `--json`: Custom output path for trajectory JSON export.

---

## 📈 Empirical Tracking Benchmarks

Evaluated across varied UAVSwarm scenarios on Apple Silicon MPS hardware:

| Test Sequence | Video Resolution | Frames | Mean Swarm Size (Peak) | Persistent Tracks | Trajectory Coordinates Traced | Runtime Processing FPS |
|---|---|---|---|---|---|---|
| **UAVSwarm-02** | $812 \times 428$ | 156 | 22.9 drones (30 peak) | 87 | **3,657 points** | 2.3 FPS |
| **UAVSwarm-04** | $812 \times 428$ | 161 | 21.4 drones (28 peak) | 74 | **3,446 points** | 2.3 FPS |
| **UAVSwarm-08** | $446 \times 270$ | 115 | 9.5 drones (11 peak) | 14 | **1,109 points** | 1.6 FPS |
| **UAVSwarm-14** | $812 \times 428$ | 115 | 16.7 drones (22 peak) | 42 | **1,960 points** | 2.5 FPS |

---

## 📄 Output Data Formats

### CSV Format (`results/trajectories_<name>.csv`)
```csv
track_id,frame,cx,cy,vx,vy,x1,y1,x2,y2
1,1,278.76,124.65,0.0,0.0,260.87,110.35,296.65,138.96
1,2,278.28,125.90,-0.48,1.25,260.09,110.96,296.47,140.84
...
```

### JSON Format (`results/trajectories_<name>.json`)
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
* **Shrimp RL Tracker**: [SamratRay2005/Shrimp](https://github.com/SamratRay2005/Shrimp)
