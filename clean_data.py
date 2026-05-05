import pandas as pd
import numpy as np

# 1. Load your raw annotated data
INPUT_CSV = 'rl_ground_truth.csv'
OUTPUT_CSV = 'rl_clean_training_data.csv'

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# 2. Extract frame numbers for mathematical sorting
df['frame_num'] = df['frame'].str.extract(r'frame_(\d+)').astype(int)
df = df.sort_values(by=['track_id', 'frame_num']).reset_index(drop=True)

# 3. Filtering Logic
valid_track_ids = []

for track_id, group in df.groupby('track_id'):
    frames = group['frame_num'].values
    cxs = group['cx'].values
    cys = group['cy'].values
    
    # Condition A: Minimum Length (At least 16 frames as required by the paper)
    if len(group) < 16:
        continue
        
    # Condition B: Zero Gaps (Strict continuous time steps)
    frame_diffs = np.diff(frames)
    if np.sum(frame_diffs > 1) > 0:
        continue
        
    # Condition C: Zero Teleportation (No jumps > 200 pixels)
    dists = np.sqrt(np.diff(cxs)**2 + np.diff(cys)**2)
    max_jump = np.max(dists) if len(dists) > 0 else 0
    if max_jump > 200:
        continue
        
    # If it passes all 3 conditions, it is a "Perfect Trajectory"
    valid_track_ids.append(track_id)

# 4. Extract only the perfect data
clean_df = df[df['track_id'].isin(valid_track_ids)]

# Drop the temporary frame_num column and save
clean_df = clean_df.drop(columns=['frame_num'])
clean_df.to_csv(OUTPUT_CSV, index=False)

print("\n=== CLEANING COMPLETE ===")
print(f"Raw Tracks Processed: {df['track_id'].nunique()}")
print(f"Perfect Tracks Saved: {len(valid_track_ids)}")
print(f"Total Clean Rows: {len(clean_df)}")
print(f"Saved to: {OUTPUT_CSV}")