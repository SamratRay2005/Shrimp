import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
import pandas as pd

# ===================== CONFIGURATION =====================
IMG_DIR = "1.gt_img"
OUTPUT_CSV = "rl_ground_truth.csv"

# Model Config
INPUT_W = 640
INPUT_H = 640
SCORE_THRES = 0.25
NMS_IOU = 0.30
MODEL_PATH = "last.onnx"
TRY_COREML = True 

# Globals for UI State
current_detections = []
detection_cache = {}  # Caches ONNX outputs so rewinding is instant
current_frame_name = ""
img_display = None
current_track_id = 1  

# ===================== LOGIC: NMS =====================
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

# ===================== INFERENCE =====================
def get_model_detections(session, input_names, output_names, frame, frame_name):
    if frame_name in detection_cache:
        return detection_cache[frame_name]

    H_orig, W_orig = frame.shape[:2]
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]
    orig_size = np.array([[float(W_orig), float(H_orig)]], dtype=np.float32)

    outputs = session.run(output_names, {
        input_names[0]: input_data, 
        input_names[1]: orig_size
    })
    
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
    
    formatted_detections = []
    if keep_indices:
        for k in keep_indices:
            bx1, by1, bx2, by2 = raw_boxes[k]
            formatted_detections.append({
                'x1': float(bx1), 'y1': float(by1), 
                'x2': float(bx2), 'y2': float(by2),
                'cx': (bx1 + bx2) / 2.0,
                'cy': (by1 + by2) / 2.0,
                'score': raw_scores[k]
            })
            
    detection_cache[frame_name] = formatted_detections
    return formatted_detections

# ===================== CSV HELPER =====================
def get_saved_annotations(frame_name):
    if not os.path.exists(OUTPUT_CSV):
        return []
    try:
        df = pd.read_csv(OUTPUT_CSV)
        frame_data = df[df['frame'] == frame_name]
        return frame_data.to_dict('records')
    except pd.errors.EmptyDataError:
        return []

# ===================== UI & INTERACTION =====================
def mouse_callback(event, x, y, flags, param):
    global img_display, current_track_id
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not current_detections:
            print("[WARN] No detections on this frame.")
            return

        min_dist = float('inf')
        closest_det = None
        
        for det in current_detections:
            dist = np.sqrt((x - det['cx'])**2 + (y - det['cy'])**2)
            if dist < min_dist:
                min_dist = dist
                closest_det = det

        if closest_det:
            row_data = {
                'frame': current_frame_name,
                'track_id': current_track_id,
                'cx': closest_det['cx'],
                'cy': closest_det['cy'],
                'x1': closest_det['x1'],
                'y1': closest_det['y1'],
                'x2': closest_det['x2'],
                'y2': closest_det['y2'],
                'score': closest_det['score']
            }
            df = pd.DataFrame([row_data])
            df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
            
            print(f"[SUCCESS] Saved ID {current_track_id} at ({closest_det['cx']:.1f}, {closest_det['cy']:.1f})")
            
            cv2.rectangle(img_display, 
                          (int(closest_det['x1']), int(closest_det['y1'])), 
                          (int(closest_det['x2']), int(closest_det['y2'])), 
                          (0, 255, 0), 2)
            cv2.putText(img_display, f"ID: {current_track_id}", 
                        (int(closest_det['x1']), max(0, int(closest_det['y1'])-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Shrimp Annotator", img_display)

# ===================== MAIN LOOP =====================
def main():
    global current_detections, current_frame_name, img_display, current_track_id
    
    print("🚀 Initializing ONNX Runtime...")
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

    cv2.namedWindow("Shrimp Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Shrimp Annotator", mouse_callback)

    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg"))) 
    if not image_paths:
        print(f"❌ No images found in {IMG_DIR}!")
        return

    print("\n=== SCRUBBING INSTRUCTIONS ===")
    print(" [CLICK] Save the active Tracking ID to the clicked shrimp.")
    print(" [ n ]   Next Frame")
    print(" [ b ]   Previous Frame (Back)")
    print(" [ r ]   Rewind to Frame 0")
    print(" [ j ]   Jump to a specific frame (Types directly on image)")
    print(" [ +/= ] Increment Tracking ID (Start tracking a new shrimp)")
    print(" [ -/_ ] Decrement Tracking ID")
    print(" [ q ]   Quit")
    print("==============================\n")

    frame_index = 0
    total_frames = len(image_paths)

    while frame_index < total_frames:
        img_path = image_paths[frame_index]
        current_frame_name = os.path.basename(img_path)
        
        frame = cv2.imread(img_path)
        if frame is None: 
            frame_index += 1
            continue
            
        img_display = frame.copy()
        
        current_detections = get_model_detections(session, input_names, output_names, frame, current_frame_name)
        
        for det in current_detections:
            cv2.rectangle(img_display, 
                          (int(det['x1']), int(det['y1'])), 
                          (int(det['x2']), int(det['y2'])), 
                          (0, 0, 255), 1)

        saved_annots = get_saved_annotations(current_frame_name)
        for ann in saved_annots:
            cv2.rectangle(img_display, 
                          (int(ann['x1']), int(ann['y1'])), 
                          (int(ann['x2']), int(ann['y2'])), 
                          (0, 255, 0), 2)
            cv2.putText(img_display, f"ID: {ann['track_id']}", 
                        (int(ann['x1']), max(0, int(ann['y1'])-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(img_display, f"Frame: {frame_index}/{total_frames-1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_display, f"ACTIVE ID: {current_track_id}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Shrimp Annotator", img_display)
        
        # Wait for user navigation input
        while True:
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('n'):     # NEXT
                frame_index += 1
                break
            elif key == ord('b'):   # BACK
                frame_index = max(0, frame_index - 1)
                break
            elif key == ord('r'):   # REWIND
                frame_index = 0
                break
            
            # FIXED: Broadened key checks for Increment/Decrement
            elif key in [ord('='), ord('+')]:   # INCREMENT ID (+)
                current_track_id += 1
                print(f"[*] Changed Active ID to: {current_track_id}")
                break 
            elif key in [ord('-'), ord('_')]:   # DECREMENT ID (-)
                current_track_id = max(1, current_track_id - 1)
                print(f"[*] Changed Active ID to: {current_track_id}")
                break 
                
            # FIXED: In-window overlay for jumping frames (Removes input() blocking)
            elif key == ord('j'):   
                jump_str = ""
                while True:
                    jump_overlay = img_display.copy()
                    cv2.putText(jump_overlay, f"JUMP TO FRAME: {jump_str}_", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    cv2.imshow("Shrimp Annotator", jump_overlay)
                    
                    k = cv2.waitKey(0) & 0xFF
                    if k == 13 or k == 10:  # Enter key
                        break
                    elif k == 27:  # Esc key cancels jump
                        jump_str = ""
                        break
                    elif k == 8 or k == 127:  # Backspace
                        jump_str = jump_str[:-1]
                    elif chr(k).isdigit():
                        jump_str += chr(k)
                        
                if jump_str.isdigit():
                    target_idx = int(jump_str)
                    if 0 <= target_idx < total_frames:
                        frame_index = target_idx
                        break
                    else:
                        print(f"[WARN] Frame {target_idx} is out of bounds (0-{total_frames-1}).")
                        break
                else:
                    break # Cancelled or empty

            elif key == ord('q') or key == 27:  
                print("\n[INFO] Exiting...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()