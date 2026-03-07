import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import math
import time
import os

# ===================== CONFIGURATION =====================
INPUT_W = 640
INPUT_H = 640
SCORE_THRES = 0.25
NMS_IOU = 0.30
DET_EVERY_N = 1

MODEL_PATH = "last.onnx"
VIDEO_PATH = "IMG_9386.mp4"
POLICY_PATH = "shrimp_tracker_policy.pth"

TRY_COREML = True 
GATE_RADIUS = 150.0

# ===================== Q-NETWORK DEFINITION =====================
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

# ===================== KALMAN FILTER =====================
class SimpleKF:
    """Predicts shrimp motion to set the Tracking Gate center."""
    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0, 0], dtype=np.float32) 
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.F = np.array([[1, 0, 1, 0], 
                           [0, 1, 0, 1], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 5.0
        self.Q = np.eye(4, dtype=np.float32) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, cx, cy):
        z = np.array([cx, cy], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

# ===================== HYSTERESIS LINE COUNTER =====================
class VectorHysteresisCounter:
    def __init__(self, ptA, ptB, buffer_width=30.0):
        self.A = np.array(ptA, dtype=np.float32)
        self.B = np.array(ptB, dtype=np.float32)
        self.buffer_width = buffer_width
        self.AB = self.B - self.A
        self.length = np.linalg.norm(self.AB)
        
        self.up = 0
        self.down = 0

    def get_state(self, center):
        """Returns 0 (Side A), 1 (Inside Buffer), or 2 (Side B)"""
        AP = center - self.A
        cross_z = self.AB[0] * AP[1] - self.AB[1] * AP[0]
        sd = cross_z / self.length # Signed perpendicular distance
        
        if sd > self.buffer_width: return 0
        elif sd < -self.buffer_width: return 2
        else: return 1

    def draw(self, img):
        A_int, B_int = tuple(self.A.astype(int)), tuple(self.B.astype(int))
        cv2.line(img, A_int, B_int, (0, 255, 255), 2)
        
        # Draw the parallel buffer zone lines
        N = np.array([-self.AB[1], self.AB[0]]) / self.length
        A1, B1 = self.A + N * self.buffer_width, self.B + N * self.buffer_width
        A2, B2 = self.A - N * self.buffer_width, self.B - N * self.buffer_width
        
        cv2.line(img, tuple(A1.astype(int)), tuple(B1.astype(int)), (0, 100, 100), 1, cv2.LINE_AA)
        cv2.line(img, tuple(A2.astype(int)), tuple(B2.astype(int)), (0, 100, 100), 1, cv2.LINE_AA)
        
        cv2.putText(img, f"Up: {self.up}  Down: {self.down}", (12, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ===================== RL TRACKER CORE =====================
class RLTracker:
    def __init__(self, device, counter):
        self.device = device
        self.counter = counter
        
        self.policy = QNetwork().to(self.device)
        self.policy.load_state_dict(torch.load(POLICY_PATH, map_location=self.device, weights_only=True))
        self.policy.eval()
        
        self.tracks = {} 
        self.next_id = 1

    def get_subregion(self, dx, dy):
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        if distance > GATE_RADIUS: return -1 
        ring = 0 if distance <= (GATE_RADIUS / 2.0) else 1
        if -math.pi/4 <= angle < math.pi/4: quadrant = 0
        elif math.pi/4 <= angle < 3*math.pi/4: quadrant = 1
        elif angle >= 3*math.pi/4 or angle < -3*math.pi/4: quadrant = 2
        else: quadrant = 3
        return (ring * 4) + quadrant

    def process_frame(self, detections):
        unmatched_dets = detections.copy()
        active_tracks = []
        
        for trk_id, trk_data in list(self.tracks.items()):
            kf = trk_data['kf']
            pred_cx, pred_cy = kf.predict()
            
            candidates = []
            for idx, det in enumerate(unmatched_dets):
                dx = det['cx'] - pred_cx
                dy = det['cy'] - pred_cy
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist <= GATE_RADIUS:
                    region_id = self.get_subregion(dx, dy)
                    candidates.append({
                        'det_idx': idx, 'dx': dx, 'dy': dy, 'region_id': region_id
                    })
            
            if candidates:
                features = [[c['dx']/GATE_RADIUS, c['dy']/GATE_RADIUS, c['region_id']/7.0] for c in candidates]
                tensor_features = torch.FloatTensor(features).to(self.device)
                
                with torch.no_grad():
                    q_values = self.policy(tensor_features).cpu().numpy().flatten()
                
                best_cand_idx = np.argmax(q_values)
                best_det_idx = candidates[best_cand_idx]['det_idx']
                best_det = unmatched_dets[best_det_idx]
                
                kf.update(best_det['cx'], best_det['cy'])
                trk_data['missed'] = 0
                trk_data['box'] = best_det['box']
                
                # 3-State Hysteresis Update
                new_state = self.counter.get_state(np.array([best_det['cx'], best_det['cy']]))
                old_state = trk_data['state']
                
                if old_state == 0 and new_state == 2: self.counter.down += 1
                elif old_state == 1 and new_state == 2: self.counter.down += 1
                elif old_state == 2 and new_state == 0: self.counter.up += 1
                elif old_state == 1 and new_state == 0: self.counter.up += 1
                    
                if new_state != 1 or old_state == -1:
                    trk_data['state'] = new_state
                
                active_tracks.append((trk_id, best_det['box']))
                unmatched_dets.pop(best_det_idx)
            else:
                trk_data['missed'] += 1
                if trk_data['missed'] > 5: 
                    del self.tracks[trk_id]
                    
        for det in unmatched_dets:
            self.tracks[self.next_id] = {
                'kf': SimpleKF(det['cx'], det['cy']),
                'state': self.counter.get_state(np.array([det['cx'], det['cy']])),
                'missed': 0,
                'box': det['box']
            }
            active_tracks.append((self.next_id, det['box']))
            self.next_id += 1
            
        return active_tracks

# ===================== Logic: NMS =====================
def nms_indices(boxes, scores, score_thr, iou_thr):
    valid_indices = [i for i, s in enumerate(scores) if s >= score_thr]
    valid_indices.sort(key=lambda i: scores[i], reverse=True)
    keep, removed = [], [False] * len(valid_indices)
    
    for i in range(len(valid_indices)):
        if removed[i]: continue
        idx_a = valid_indices[i]
        keep.append(idx_a)
        box_a = boxes[idx_a]
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        
        for j in range(i + 1, len(valid_indices)):
            if removed[j]: continue
            idx_b = valid_indices[j]
            box_b = boxes[idx_b]
            xx1, yy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            xx2, yy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
            inter = w * h
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union = area_a + area_b - inter + 1e-6
            if (inter / union) > iou_thr: removed[j] = True
    return keep

# ===================== MAIN LOOP =====================
def main():
    print("🚀 Initializing ONNX & RL Policy on Apple Silicon...")
    
    # 1. Hardware Targeting
    rl_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    providers = ['CPUExecutionProvider']
    if TRY_COREML:
        coreml_options = {'MLComputeUnits': 'CPU_AND_GPU'}
        providers = [('CoreMLExecutionProvider', coreml_options), 'CPUExecutionProvider']

    try:
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception as e:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    
    # 2. Init Pipeline
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return print("❌ Error opening video.")

    counter = VectorHysteresisCounter((450, 303), (480, 1836), buffer_width=40.0)
    tracker = RLTracker(rl_device, counter)
    
    frame_idx, overall_time = 0, 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_idx += 1
        H_orig, W_orig = frame.shape[:2]
        
        if frame_idx % DET_EVERY_N == 0:
            resized = cv2.resize(frame, (INPUT_W, INPUT_H))
            input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]
            orig_size = np.array([[float(W_orig), float(H_orig)]], dtype=np.float32)

            outputs = session.run(output_names, {input_names[0]: input_data, input_names[1]: orig_size})
            
            pred_boxes = np.squeeze(outputs[1]) 
            pred_scores = np.squeeze(outputs[2]) 
            if pred_scores.ndim == 0: pred_scores, pred_boxes = np.array([pred_scores]), np.array([pred_boxes])

            raw_boxes, raw_scores = [], []
            for i, score in enumerate(pred_scores):
                x1, y1, x2, y2 = pred_boxes[i]
                if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                    raw_boxes.append([x1, y1, x2, y2])
                    raw_scores.append(float(score))
            
            keep_indices = nms_indices(raw_boxes, raw_scores, SCORE_THRES, NMS_IOU)
            
            # Format detections for the RL Tracker
            detections = []
            for k in keep_indices:
                x1, y1, x2, y2 = raw_boxes[k]
                detections.append({'cx': (x1+x2)/2, 'cy': (y1+y2)/2, 'box': [x1,y1,x2,y2]})
        else:
            detections = []

        # Tracker Update
        t_start = time.time()
        active_tracks = tracker.process_frame(detections)
        overall_time += (time.time() - t_start) * 1000

        # UI Drawing
        for trk_id, box in active_tracks:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (3, 155, 229), 2)
            cv2.putText(frame, f"ID: {trk_id}", (int(x1), max(0, int(y1)-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (115, 115, 229), 2, cv2.LINE_AA)

        counter.draw(frame)
        cv2.imshow("M1 RL Shrimp Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    avg_fps = 1000.0 / (overall_time / frame_idx) if frame_idx > 0 else 0
    print(f"\n✅ Final Counts -> Up: {counter.up}, Down: {counter.down}")
    print(f"Avg Tracker FPS: {int(avg_fps)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()