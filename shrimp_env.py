import pandas as pd
import numpy as np
import math

class ShrimpTrackingEnv:
    def __init__(self, csv_path, gate_radius=100, num_clutter=4):
        print(f"Loading Ground Truth from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.track_ids = self.df['track_id'].unique()
        
        self.gate_radius = gate_radius
        self.num_clutter = num_clutter # Number of fake detections to generate per frame
        
        # Internal state variables
        self.current_track = None
        self.current_frame_idx = 0
        self.track_data = None
        self.true_measurement = None

    def reset(self):
        """Starts a new episode by picking a random perfect trajectory."""
        track_id = np.random.choice(self.track_ids)
        self.track_data = self.df[self.df['track_id'] == track_id].reset_index(drop=True)
        self.current_frame_idx = 0
        
        # The initial position (Frame 0) is known. We step to Frame 1 to make the first prediction.
        return self.step()

    def generate_clutter(self, cx, cy):
        """Generates random fake measurements within the tracking gate."""
        fake_measurements = []
        for _ in range(self.num_clutter):
            # Random angle and distance within the gate
            angle = np.random.uniform(0, 2 * math.pi)
            distance = np.random.uniform(0, self.gate_radius)
            
            fake_x = cx + distance * math.cos(angle)
            fake_y = cy + distance * math.sin(angle)
            fake_measurements.append((fake_x, fake_y))
            
        return fake_measurements

    def get_subregion(self, dx, dy):
        """
        Maps a distance (dx, dy) into one of the 8 tracking gate subregions 
        defined in the Qu et al. paper.
        """
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx) # Range: -pi to pi
        
        # If outside the gate, it's irrelevant (State 0 or ignored)
        if distance > self.gate_radius:
            return -1 
            
        # Determine Inner (0) vs Outer (1) ring
        ring = 0 if distance <= (self.gate_radius / 2.0) else 1
        
        # Determine Quadrant (0 to 3)
        if -math.pi/4 <= angle < math.pi/4:
            quadrant = 0 # Right
        elif math.pi/4 <= angle < 3*math.pi/4:
            quadrant = 1 # Top
        elif angle >= 3*math.pi/4 or angle < -3*math.pi/4:
            quadrant = 2 # Left
        else:
            quadrant = 3 # Bottom
            
        # Unique Subregion ID (0 to 7)
        return (ring * 4) + quadrant

    def step(self):
        """Advances the trajectory by one frame and builds the State."""
        self.current_frame_idx += 1
        
        # Check if we reached the end of the trajectory
        if self.current_frame_idx >= len(self.track_data):
            return None, None, None, True # done = True
            
        # 1. Get the previous known position (Acts as our predicted center for the gate)
        prev_row = self.track_data.iloc[self.current_frame_idx - 1]
        pred_cx, pred_cy = prev_row['cx'], prev_row['cy']
        
        # 2. Get the True Measurement for the current frame
        curr_row = self.track_data.iloc[self.current_frame_idx]
        true_cx, true_cy = curr_row['cx'], curr_row['cy']
        
        # 3. Generate Clutter around the true measurement
        candidates = self.generate_clutter(pred_cx, pred_cy)
        candidates.append((true_cx, true_cy)) # Mix the truth with the fakes
        np.random.shuffle(candidates) # Shuffle so truth isn't always last
        
        # Find which index contains the truth so we can calculate reward later
        true_index = candidates.index((true_cx, true_cy))
        
        # 4. Build the State Matrix (S_t)
        # We categorize each candidate into one of the 8 subregions
        state_representation = []
        for (cand_x, cand_y) in candidates:
            dx = cand_x - pred_cx
            dy = cand_y - pred_cy
            region_id = self.get_subregion(dx, dy)
            
            # Save the features the agent will use to make a decision
            state_representation.append({
                'dx': dx,
                'dy': dy,
                'region_id': region_id
            })
            
        return state_representation, candidates, true_index, False

# ==========================================
# TEST THE ENVIRONMENT
# ==========================================
if __name__ == "__main__":
    # Initialize the Environment
    env = ShrimpTrackingEnv(csv_path='rl_clean_training_data.csv', gate_radius=150, num_clutter=4)
    
    # Start a random trajectory
    state, candidates, true_index, done = env.reset()
    
    print("\n=== RL ENVIRONMENT INITIALIZED ===")
    print(f"Tracking Gate Radius: {env.gate_radius} pixels")
    print(f"Clutter Level: {env.num_clutter} fake measurements added")
    
    print("\n--- STATE t (What the Agent Sees) ---")
    for i, st in enumerate(state):
        truth_label = "<-- [GROUND TRUTH]" if i == true_index else ""
        print(f"Candidate {i}: Region [{st['region_id']}] | dx: {st['dx']:7.2f}, dy: {st['dy']:7.2f} {truth_label}")
        
    print("\n--- ACTION & REWARD LOGIC ---")
    print(f"If Agent selects Action {true_index} -> Reward: +1")
    print(f"If Agent selects any other Action -> Reward: -1\n")