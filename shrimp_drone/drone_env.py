"""
Drone Tracking Environment — Adapted from Shrimp Architecture
===============================================================
Original: ShrimpTrackingEnv (shrimp/shrimp_env.py)
Adapted:  DroneTrackingEnv for UAVSwarm multi-drone detection + tracking

KEY ARCHITECTURAL CHANGES vs SHRIMP:
─────────────────────────────────────
1. DATA SOURCE:
   - Shrimp: single CSV with columns [frame, track_id, cx, cy, x1, y1, x2, y2, score]
   - Drone:  UAVSwarm MOT16 gt.txt → converted to same CSV format per sequence
   - Multiple simultaneous tracks (3-23 drones) instead of one-at-a-time shrimp

2. GATE RADIUS:
   - Shrimp: 150px (shrimp move slowly in a tank with a fixed camera)
   - Drone:  250px (drones move fast; aerial perspective = large pixel displacement)

3. CLUTTER MODEL:
   - Shrimp: 4 random fake measurements (background noise)
   - Drone:  4 fakes + OTHER DRONES as real but wrong candidates
   - This makes the data association problem much harder — correct drone in a swarm

4. STATE FEATURES (5D per candidate, same as Shrimp v2):
   - dx, dy: displacement from predicted gate center (normalized by gate_radius)
   - region_id: 8-subregion spatial encoding (inner/outer ring × 4 quadrants)
   - vx, vy: velocity of the TRACK (not candidate) — helps match moving drone

5. REWARD FUNCTION (enhanced from Shrimp):
   - Shrimp: +1 / -1 binary
   - Drone:  +1 exact hit / +0.5 near miss (< 20% gate) / -1 wrong → smoother gradients
   - WHY: Drone swarms have very close neighbors; near-miss shouldn't be penalized equally

6. EPISODE STRUCTURE:
   - Shrimp: one track per episode
   - Drone:  one track per episode, but training data comes from multi-drone sequences
   - This means the model sees clutter that is REAL drone detections from other tracks

USAGE:
  from drone_env import DroneTrackingEnv
  env = DroneTrackingEnv('drone_training_data.csv', gate_radius=250, num_clutter=4)
  state, candidates, true_idx, done = env.reset()
"""

import pandas as pd
import numpy as np
import math


class DroneTrackingEnv:
    """
    RL Environment for data association in drone swarm tracking.

    The agent's job at each timestep: given N candidate detections within the
    tracking gate, pick the one that corresponds to the tracked drone.

    State per candidate: (dx, dy, region_id, vx, vy)
    - dx, dy     : offset from Kalman-predicted position
    - region_id  : spatial zone (0-7) in the 8-subregion gate decomposition
    - vx, vy     : velocity of the current track (from previous two observations)

    Reward: +1.0 (exact), +0.5 (near miss < 20% gate), -1.0 (wrong)
    """

    def __init__(self, csv_path: str, gate_radius: float = 250.0,
                 num_clutter: int = 4, use_swarm_clutter: bool = True):
        """
        Args:
            csv_path:          Path to training CSV (frame, track_id, cx, cy, ...)
            gate_radius:       Tracking gate size in pixels
            num_clutter:       Number of pure random fake measurements per frame
            use_swarm_clutter: If True, also add other drones from same frame as
                               real-but-wrong candidates (makes problem harder/real)
        """
        print(f"[DroneEnv] Loading training data from {csv_path}...")
        self.df = pd.read_csv(csv_path)

        # Required columns: frame, track_id, cx, cy
        required = {'frame', 'track_id', 'cx', 'cy'}
        assert required.issubset(self.df.columns), \
            f"CSV must have columns: {required}. Got: {set(self.df.columns)}"

        self.track_ids = self.df['track_id'].unique()
        self.gate_radius = gate_radius
        self.num_clutter = num_clutter
        self.use_swarm_clutter = use_swarm_clutter

        # Build frame→all detections lookup for swarm clutter
        if use_swarm_clutter:
            self._frame_lookup = (
                self.df.groupby('frame')[['cx', 'cy', 'track_id']]
                .apply(lambda g: g.to_dict('records'))
                .to_dict()
            )

        # Internal state
        self.track_data = None
        self.current_frame_idx = 0
        self.current_track_id = None

        print(f"[DroneEnv] Ready: {len(self.track_ids)} tracks, "
              f"gate={gate_radius}px, clutter={num_clutter}, "
              f"swarm_clutter={use_swarm_clutter}")

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        """Start a new episode on a randomly selected drone track."""
        self.current_track_id = np.random.choice(self.track_ids)
        self.track_data = (
            self.df[self.df['track_id'] == self.current_track_id]
            .reset_index(drop=True)
        )
        self.current_frame_idx = 0
        return self.step()

    # ── Subregion Encoding ───────────────────────────────────────────────────

    def get_dynamic_gate(self, vx: float = 0.0, vy: float = 0.0, missed: int = 0) -> float:
        """
        Velocity-Adaptive Dynamic Gating:
        Scales tracking gate dynamically with drone speed and prediction uncertainty.
        - Hovering/slow formation drones: gate shrinks to ~20px (prevents neighbor swarm stealing).
        - High-speed dashes: gate expands up to 65px (prevents losing accelerating drones).
        """
        speed = math.sqrt(vx * vx + vy * vy)
        dynamic_r = 20.0 + (2.0 * speed) + (4.0 * missed)
        return float(np.clip(dynamic_r, 20.0, 65.0))

    def get_subregion(self, dx: float, dy: float, gate_r: float = None) -> int:
        """
        Map (dx, dy) offset into one of 8 gate subregions.
        Returns -1 if outside gate (should not appear in candidates).

        Subregion layout (same as Shrimp paper, Qu et al.):
          Ring 0 (inner, r ≤ gate/2): subregions 0-3
          Ring 1 (outer, r > gate/2): subregions 4-7
          Quadrants: 0=Right, 1=Top, 2=Left, 3=Bottom
        """
        R = gate_r if gate_r is not None else self.gate_radius
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > R:
            return -1

        ring = 0 if dist <= (R / 2.0) else 1
        angle = math.atan2(dy, dx)

        if -math.pi / 4 <= angle < math.pi / 4:
            quadrant = 0   # Right
        elif math.pi / 4 <= angle < 3 * math.pi / 4:
            quadrant = 1   # Top
        elif angle >= 3 * math.pi / 4 or angle < -3 * math.pi / 4:
            quadrant = 2   # Left
        else:
            quadrant = 3   # Bottom

        return (ring * 4) + quadrant

    # ── Clutter Generation ───────────────────────────────────────────────────

    def _random_clutter(self, pred_cx: float, pred_cy: float) -> list:
        """Generate pure random fake detections within the gate."""
        fakes = []
        for _ in range(self.num_clutter):
            angle = np.random.uniform(0, 2 * math.pi)
            dist = np.random.uniform(0, self.gate_radius)
            fakes.append((
                pred_cx + dist * math.cos(angle),
                pred_cy + dist * math.sin(angle),
            ))
        return fakes

    def _swarm_clutter(self, frame_name: str, pred_cx: float, pred_cy: float,
                       true_cx: float, true_cy: float) -> list:
        """
        Get OTHER drones from the same frame as real-but-wrong candidates.
        This is the key difference from shrimp: in a drone swarm, the 'noise'
        is actually other real drones — much harder to reject than random clutter.
        """
        if frame_name not in self._frame_lookup:
            return []

        swarm = []
        for det in self._frame_lookup[frame_name]:
            if det['track_id'] == self.current_track_id:
                continue  # Skip the target itself
            cx, cy = det['cx'], det['cy']
            dx = cx - pred_cx
            dy = cy - pred_cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= self.gate_radius:
                swarm.append((cx, cy))

        return swarm

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self):
        """
        Advance one frame and build the candidate state.

        Returns:
            state_representation: list of dicts with (dx, dy, region_id, vx, vy)
            candidates:           list of (cx, cy) tuples
            true_index:           index of the true detection in candidates
            done:                 True if episode finished
        """
        self.current_frame_idx += 1

        if self.current_frame_idx >= len(self.track_data):
            return None, None, None, True  # Episode over

        prev_row = self.track_data.iloc[self.current_frame_idx - 1]
        pred_cx, pred_cy = float(prev_row['cx']), float(prev_row['cy'])

        # Velocity estimate (from previous two frames when available)
        if self.current_frame_idx >= 2:
            pp_row = self.track_data.iloc[self.current_frame_idx - 2]
            vx = float(prev_row['cx']) - float(pp_row['cx'])
            vy = float(prev_row['cy']) - float(pp_row['cy'])
        else:
            vx, vy = 0.0, 0.0

        curr_row = self.track_data.iloc[self.current_frame_idx]
        true_cx = float(curr_row['cx'])
        true_cy = float(curr_row['cy'])
        frame_name = str(curr_row.get('frame', ''))

        # Build candidate pool:
        # 1. Pure random clutter (background noise)
        candidates = self._random_clutter(pred_cx, pred_cy)

        # 2. Real swarm clutter (other drones from same frame — key for drones)
        if self.use_swarm_clutter:
            swarm_cands = self._swarm_clutter(frame_name, pred_cx, pred_cy,
                                              true_cx, true_cy)
            candidates.extend(swarm_cands)

        # 3. Ground truth detection (the one the agent must find)
        candidates.append((true_cx, true_cy))

        # Shuffle so the true index is not always last
        np.random.shuffle(candidates)
        true_index = candidates.index((true_cx, true_cy))

        # Build state representation
        state = []
        for (cx, cy) in candidates:
            dx = cx - pred_cx
            dy = cy - pred_cy
            state.append({
                'dx':        dx,
                'dy':        dy,
                'region_id': self.get_subregion(dx, dy),
                'vx':        vx,
                'vy':        vy,
            })

        return state, candidates, true_index, False


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'drone_training_data.csv'
    env = DroneTrackingEnv(csv, gate_radius=250, num_clutter=4)

    state, candidates, true_idx, done = env.reset()
    print(f"\n=== DroneTrackingEnv Test ===")
    print(f"Gate radius: {env.gate_radius}px | Clutter: {env.num_clutter}")
    print(f"\nFrame state ({len(candidates)} candidates):")
    for i, s in enumerate(state):
        tag = " ← GROUND TRUTH" if i == true_idx else ""
        print(f"  [{i}] region={s['region_id']} | "
              f"dx={s['dx']:+6.1f} dy={s['dy']:+6.1f} | "
              f"vx={s['vx']:+5.1f} vy={s['vy']:+5.1f}{tag}")
    print(f"\nIf agent picks {true_idx} → reward +1.0")
