import pandas as pd
import numpy as np
import math


class ShrimpTrackingEnv:
    def __init__(self, csv_path, gate_radius=100, num_clutter=4):
        print(f"Loading Ground Truth from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.track_ids = self.df['track_id'].unique()

        self.gate_radius = gate_radius
        self.num_clutter = num_clutter

        self.current_track = None
        self.current_frame_idx = 0
        self.track_data = None

    def reset(self):
        track_id = np.random.choice(self.track_ids)
        self.track_data = self.df[self.df['track_id']
                                  == track_id].reset_index(drop=True)
        self.current_frame_idx = 0
        return self.step()

    def generate_clutter(self, cx, cy):
        fake_measurements = []
        for _ in range(self.num_clutter):
            angle = np.random.uniform(0, 2 * math.pi)
            distance = np.random.uniform(0, self.gate_radius)
            fake_x = cx + distance * math.cos(angle)
            fake_y = cy + distance * math.sin(angle)
            fake_measurements.append((fake_x, fake_y))
        return fake_measurements

    def get_subregion(self, dx, dy):
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)

        if distance > self.gate_radius:
            return -1

        ring = 0 if distance <= (self.gate_radius / 2.0) else 1

        if -math.pi/4 <= angle < math.pi/4:
            quadrant = 0
        elif math.pi/4 <= angle < 3*math.pi/4:
            quadrant = 1
        elif angle >= 3*math.pi/4 or angle < -3*math.pi/4:
            quadrant = 2
        else:
            quadrant = 3

        return (ring * 4) + quadrant

    def step(self):
        self.current_frame_idx += 1

        if self.current_frame_idx >= len(self.track_data):
            return None, None, None, True

        prev_row = self.track_data.iloc[self.current_frame_idx - 1]
        pred_cx, pred_cy = prev_row['cx'], prev_row['cy']

        # Velocity
        if self.current_frame_idx >= 2:
            prev_prev_row = self.track_data.iloc[self.current_frame_idx - 2]
            vx = prev_row['cx'] - prev_prev_row['cx']
            vy = prev_row['cy'] - prev_prev_row['cy']
        else:
            vx, vy = 0.0, 0.0

        curr_row = self.track_data.iloc[self.current_frame_idx]
        true_cx, true_cy = curr_row['cx'], curr_row['cy']

        candidates = self.generate_clutter(pred_cx, pred_cy)
        candidates.append((true_cx, true_cy))
        np.random.shuffle(candidates)

        true_index = candidates.index((true_cx, true_cy))

        state_representation = []
        for (cand_x, cand_y) in candidates:
            dx = cand_x - pred_cx
            dy = cand_y - pred_cy
            region_id = self.get_subregion(dx, dy)

            state_representation.append({
                'dx': dx,
                'dy': dy,
                'region_id': region_id,
                'vx': vx,
                'vy': vy
            })

        return state_representation, candidates, true_index, False
