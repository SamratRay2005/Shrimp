import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import math

# ===================== CONFIG =====================
INPUT_W = 640
INPUT_H = 640
SCORE_THRES = 0.25
NMS_IOU = 0.30
DET_EVERY_N = 1

MODEL_PATH = "last.onnx"
VIDEO_PATH = "IMG_9386.mp4"
POLICY_PATH = "shrimp_tracker_policy.pth"

GATE_RADIUS = 150.0

# ===================== Q-NETWORK =====================


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 64)
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

# ===================== HYSTERESIS COUNTER =====================


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
        AP = center - self.A
        cross_z = self.AB[0] * AP[1] - self.AB[1] * AP[0]
        sd = cross_z / self.length

        if sd > self.buffer_width:
            return 0
        elif sd < -self.buffer_width:
            return 2
        else:
            return 1

    def draw(self, img):
        A_int, B_int = tuple(self.A.astype(int)), tuple(self.B.astype(int))
        cv2.line(img, A_int, B_int, (0, 255, 255), 2)

        # Draw buffer zone lines (THIS WAS MISSING BEFORE)
        N = np.array([-self.AB[1], self.AB[0]]) / self.length
        A1, B1 = self.A + N * self.buffer_width, self.B + N * self.buffer_width
        A2, B2 = self.A - N * self.buffer_width, self.B - N * self.buffer_width

        cv2.line(img, tuple(A1.astype(int)), tuple(
            B1.astype(int)), (0, 100, 100), 1)
        cv2.line(img, tuple(A2.astype(int)), tuple(
            B2.astype(int)), (0, 100, 100), 1)

        cv2.putText(img, f"Up: {self.up} Down: {self.down}",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ===================== RL TRACKER =====================


class RLTracker:
    def __init__(self, device, counter):
        self.device = device
        self.counter = counter

        self.policy = QNetwork().to(device)
        self.policy.load_state_dict(
            torch.load(POLICY_PATH, map_location=device))
        self.policy.eval()

        self.tracks = {}
        self.next_id = 1

    def get_subregion(self, dx, dy):
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)

        if distance > GATE_RADIUS:
            return -1

        ring = 0 if distance <= (GATE_RADIUS/2) else 1

        if -math.pi/4 <= angle < math.pi/4:
            quadrant = 0
        elif math.pi/4 <= angle < 3*math.pi/4:
            quadrant = 1
        elif angle >= 3*math.pi/4 or angle < -3*math.pi/4:
            quadrant = 2
        else:
            quadrant = 3

        return ring*4 + quadrant

    def process_frame(self, detections):
        unmatched = detections.copy()
        active = []

        for tid, data in list(self.tracks.items()):
            kf = data['kf']
            pred_cx, pred_cy = kf.predict()

            prev_cx = data.get('prev_cx', pred_cx)
            prev_cy = data.get('prev_cy', pred_cy)

            vx = pred_cx - prev_cx
            vy = pred_cy - prev_cy

            candidates = []

            for i, det in enumerate(unmatched):
                dx = det['cx'] - pred_cx
                dy = det['cy'] - pred_cy
                dist = math.sqrt(dx*dx + dy*dy)

                if dist <= GATE_RADIUS:
                    candidates.append({
                        'det_idx': i,
                        'dx': dx,
                        'dy': dy,
                        'region_id': self.get_subregion(dx, dy),
                        'vx': vx,
                        'vy': vy
                    })

            if candidates:
                feats = [[
                    c['dx']/GATE_RADIUS,
                    c['dy']/GATE_RADIUS,
                    c['region_id']/7.0,
                    c['vx']/GATE_RADIUS,
                    c['vy']/GATE_RADIUS
                ] for c in candidates]

                tensor = torch.FloatTensor(feats).to(self.device)

                with torch.no_grad():
                    q = self.policy(tensor).cpu().numpy().flatten()

                best = np.argmax(q)
                det_idx = candidates[best]['det_idx']
                det = unmatched[det_idx]

                kf.update(det['cx'], det['cy'])

                data['prev_cx'] = det['cx']
                data['prev_cy'] = det['cy']
                data['box'] = det['box']
                data['missed'] = 0

                # 🔥 FULL ORIGINAL HYSTERESIS LOGIC RESTORED
                new_state = self.counter.get_state(
                    np.array([det['cx'], det['cy']]))
                old_state = data.get('state', -1)

                if old_state == 0 and new_state == 2:
                    self.counter.down += 1
                elif old_state == 1 and new_state == 2:
                    self.counter.down += 1
                elif old_state == 2 and new_state == 0:
                    self.counter.up += 1
                elif old_state == 1 and new_state == 0:
                    self.counter.up += 1

                if new_state != 1 or old_state == -1:
                    data['state'] = new_state

                active.append((tid, det['box']))
                unmatched.pop(det_idx)
            else:
                data['missed'] += 1
                if data['missed'] > 5:
                    del self.tracks[tid]

        for det in unmatched:
            self.tracks[self.next_id] = {
                'kf': SimpleKF(det['cx'], det['cy']),
                'box': det['box'],
                'missed': 0,
                'prev_cx': det['cx'],
                'prev_cy': det['cy'],
                'state': self.counter.get_state(np.array([det['cx'], det['cy']]))
            }
            active.append((self.next_id, det['box']))
            self.next_id += 1

        return active

# ===================== MAIN =====================


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    session = ort.InferenceSession(MODEL_PATH)
    input_names = [i.name for i in session.get_inputs()]

    cap = cv2.VideoCapture(VIDEO_PATH)
    counter = VectorHysteresisCounter((450, 303), (480, 1836), 40)
    tracker = RLTracker(device, counter)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]

        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = img.transpose(2, 0, 1)[np.newaxis, :]

        orig_size = np.array([[float(W), float(H)]], dtype=np.float32)

        outputs = session.run(None, {
            input_names[0]: img,
            input_names[1]: orig_size
        })

        boxes = np.squeeze(outputs[1])
        scores = np.squeeze(outputs[2])

        detections = []
        for i, s in enumerate(scores):
            if s > SCORE_THRES:
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'cx': (x1+x2)/2,
                    'cy': (y1+y2)/2,
                    'box': [x1, y1, x2, y2]
                })

        tracks = tracker.process_frame(detections)

        for tid, box in tracks:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        counter.draw(frame)

        cv2.imshow("RL Shrimp Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ADD THESE LINES HERE:
    print("-" * 20)
    print(f"Final Count - Up: {counter.up}")
    print(f"Final Count - Down: {counter.down}")
    print("-" * 20)


if __name__ == "__main__":
    main()