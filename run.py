import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
import sys
import os
import time

# ===================== SETUP IMPORT =====================
sys.path.append(os.path.join(os.getcwd(), "OC_SORT"))

try:
    from trackers.ocsort_tracker.ocsort import OCSort
    print("✅ Successfully loaded local OC-SORT.")
except ImportError as e:
    print(f"\n[ERROR] Could not import OCSort. {e}")
    exit()

# ===================== Config =====================
INPUT_W = 640
INPUT_H = 640
SCORE_THRES = 0.25
NMS_IOU = 0.30
DET_EVERY_N = 1
MODEL_PATH = "last.onnx"
VIDEO_PATH = "IMG_9386.mp4"

# Set this to False if the Mac GPU still rejects the model's operations.
TRY_COREML = True 

# ===================== Logic: NMS =====================
def nms_indices(boxes, scores, score_thr, iou_thr):
    valid_indices = [i for i, s in enumerate(scores) if s >= score_thr]
    valid_indices.sort(key=lambda i: scores[i], reverse=True)
    
    keep = []
    removed = [False] * len(valid_indices)
    
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
            if (inter / union) > iou_thr:
                removed[j] = True
    return keep

# ===================== Logic: Line Counter =====================
class LineCounter:
    def __init__(self, ptA, ptB):
        self.A = np.array(ptA, dtype=np.float32)
        self.B = np.array(ptB, dtype=np.float32)
        self.up = 0
        self.down = 0
        self.dist_thresh = 15.0 
        self.cooldown_frames = 10
        self.state = defaultdict(lambda: {'last_sign': 0, 'last_frame': -999999})

    def update(self, track_id, center, frame_idx):
        AB = self.B - self.A
        AP = center - self.A
        length = np.linalg.norm(AB)
        if length < 1e-6: return
        cross_z = AB[0] * AP[1] - AB[1] * AP[0]
        sd = cross_z / length
        sign = 1 if sd > 0 else (-1 if sd < 0 else 0)
        ts = self.state[track_id]
        if ts['last_sign'] == 0:
            ts['last_sign'] = sign
            return
        if sign != 0 and sign != ts['last_sign']:
            if abs(sd) <= self.dist_thresh and (frame_idx - ts['last_frame']) > self.cooldown_frames:
                if ts['last_sign'] < 0 and sign > 0: self.up += 1
                elif ts['last_sign'] > 0 and sign < 0: self.down += 1
                ts['last_frame'] = frame_idx
        ts['last_sign'] = sign

    def draw(self, img):
        cv2.line(img, tuple(self.A.astype(int)), tuple(self.B.astype(int)), (0, 255, 255), 2)
        cv2.putText(img, f"Up: {self.up}  Down: {self.down}", (12, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ===================== Main =====================
def main():
    print("🚀 Initializing ONNX Runtime...")
    
    providers = ['CPUExecutionProvider']
    if TRY_COREML:
        # Crucial Fix: Bypassing the Neural Engine (ANE) to stop the 0-dimension Slice crash.
        coreml_options = {
            'MLComputeUnits': 'CPU_AND_GPU',
        }
        providers = [('CoreMLExecutionProvider', coreml_options), 'CPUExecutionProvider']

    try:
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print(f"✅ Success! Active execution providers: {session.get_providers()}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}. Falling back to pure CPU.")
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error opening video.")
        return

    tracker = OCSort(det_thresh=0.0, max_age=100, min_hits=0, iou_threshold=0.1, dist_weight=0.4)
    counter = LineCounter((450, 303), (480, 1836))
    
    frame_idx, overall_time = 0, 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            print("End of stream.")
            break
        
        frame_idx += 1
        H_orig, W_orig = frame.shape[:2]
        
        if frame_idx % DET_EVERY_N == 0:
            # Preprocessing
            resized = cv2.resize(frame, (INPUT_W, INPUT_H))
            input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]
            orig_size = np.array([[float(W_orig), float(H_orig)]], dtype=np.float32)

            # --- INFERENCE ---
            try:
                outputs = session.run(output_names, {
                    input_names[0]: input_data, 
                    input_names[1]: orig_size
                })
            except Exception as e:
                print(f"\n[FATAL ERROR] Model inference crashed on the GPU/ANE.")
                print(f"Error Details: {e}")
                print("\n👉 FIX: Set 'TRY_COREML = False' at line 28 of this script to run purely on CPU.")
                break
            
            pred_boxes = np.squeeze(outputs[1]) 
            pred_scores = np.squeeze(outputs[2]) 
            if pred_scores.ndim == 0:
                pred_scores, pred_boxes = np.array([pred_scores]), np.array([pred_boxes])

            raw_boxes, raw_scores = [], []
            for i, score in enumerate(pred_scores):
                x1, y1, x2, y2 = pred_boxes[i]
                if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                    raw_boxes.append([x1, y1, x2, y2])
                    raw_scores.append(float(score))
            
            keep_indices = nms_indices(raw_boxes, raw_scores, SCORE_THRES, NMS_IOU)
            detections_for_tracker = np.array([[*raw_boxes[k], raw_scores[k]] for k in keep_indices]) if keep_indices else np.empty((0, 5))
        else:
            detections_for_tracker = np.empty((0, 5))

        # Tracker Update
        t_start = time.time()
        tracks = tracker.update(detections_for_tracker, [H_orig, W_orig], [H_orig, W_orig])
        overall_time += (time.time() - t_start) * 1000

        for track in tracks:
            if len(track) < 5: continue
            x1, y1, x2, y2, track_id = track[:5]
            center = np.array([(x1+x2)/2, (y1+y2)/2])
            counter.update(int(track_id), center, frame_idx)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (3, 155, 229), 2)
            
            # Draw the ID label
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), max(0, int(y1)-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (115, 115, 229), 2, cv2.LINE_AA)

        counter.draw(frame)
        cv2.imshow("M1 Accelerated OC-SORT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    avg_fps = 1000.0 / (overall_time / frame_idx) if frame_idx > 0 else 0
    print(f"\nFinal Counts -> Up: {counter.up}, Down: {counter.down}")
    print(f"Avg Tracker FPS: {int(avg_fps)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()