"""
Railtech.py

Enhanced railway-track obstacle detection
- YOLOv8 (ultralytics) detector for humans/animals/vehicles
- Temporal filtering (debounce) so alerts fire only when an object persists for N frames
- SQLite persistence of alert records and snapshot paths
- Twilio SMS integration (configurable)
- Train-stop hook (placeholder implementation, supports GPIO when enabled)

Requirements:
  pip install ultralytics opencv-python requests geocoder Pillow twilio

Configure the CONFIG dictionary below before running (Twilio creds, camera index/stream, GPIO options).

Note: This script is a template. Test thoroughly before deploying near real railways. Stopping a real train requires integration with train control systems and safety approvals.
"""

import os
import time
import json
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict

import cv2
import requests
import geocoder
from PIL import Image

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics required. Install with `pip install ultralytics`") from e

# Twilio optional
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# ------------------
# Configuration
# ------------------
CONFIG = {
    "video_source": 0,
    "yolo_model": "yolov8n.pt",
    "conf_threshold": 0.35,
    # classes considered threats (map COCO names to a threat category)
    "threat_classes": {
        "person": "human",
        "dog": "animal",
        "cat": "animal",
        "cow": "animal",
        "sheep": "animal",
        "horse": "animal",
        "bicycle": "obstacle",
        "motorcycle": "vehicle",
        "car": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
    },

    # How many consecutive frames an object must be seen for an alert
    "alert_persistence_frames": 3,

    # How long (seconds) to pause between alerts of the same label/location
    "alert_cooldown_seconds": 30,

    "output_dir": "alerts",
    "database": "alerts.db",

    # Twilio (optional)
    "twilio": {
        "enabled": False,
        "account_sid": "",
        "auth_token": "",
        "from_number": "",  # +1... format
        "to_number": "",
    },

    # Webhook (optional)
    "webhook_url": "",

    # Train control options: 'mock' (print), or 'gpio' to use RPi.GPIO (set enabled True to attempt)
    "train_control": {
        "method": "mock",
        "gpio_enabled": False,
        "gpio_pin": 18,
        "stop_duration_seconds": 10,
    },

    # Whether to show OpenCV windows
    "show_preview": True,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ------------------
# Database helpers
# ------------------
def init_db(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT,
            latitude REAL,
            longitude REAL,
            method TEXT,
            label TEXT,
            threat_type TEXT,
            confidence REAL,
            bbox TEXT,
            snapshot_path TEXT
        )
        """
    )
    conn.commit()
    return conn

DB_CONN = init_db(CONFIG["database"])
DB_LOCK = threading.Lock()

def save_alert_db(record):
    with DB_LOCK:
        cur = DB_CONN.cursor()
        cur.execute(
            "INSERT INTO alerts (timestamp_utc, latitude, longitude, method, label, threat_type, confidence, bbox, snapshot_path) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record.get("timestamp_utc"),
                record.get("latitude"),
                record.get("longitude"),
                record.get("method"),
                record.get("label"),
                record.get("threat_type"),
                record.get("confidence"),
                json.dumps(record.get("bbox")),
                record.get("snapshot_path"),
            ),
        )
        DB_CONN.commit()

# ------------------
# Location helper
# ------------------
def get_location():
    try:
        g = geocoder.ip("me")
        if g.ok and g.latlng:
            return {"latitude": float(g.latlng[0]), "longitude": float(g.latlng[1]), "method": "ip"}
    except Exception:
        pass
    return {"latitude": None, "longitude": None, "method": "unknown"}

# ------------------
# Train control (placeholder)
# ------------------
def stop_train(duration_seconds=None):
    method = CONFIG["train_control"].get("method", "mock")
    duration_seconds = duration_seconds or CONFIG["train_control"].get("stop_duration_seconds", 10)

    if method == "gpio" and CONFIG["train_control"].get("gpio_enabled"):
        try:
            import RPi.GPIO as GPIO
            pin = CONFIG["train_control"].get("gpio_pin")
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            print(f"Asserting stop signal on GPIO {pin} for {duration_seconds}s")
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(duration_seconds)
            GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup()
            return True
        except Exception as e:
            print("GPIO stop_train failed:", e)
            return False
    else:
        # Mock: print and sleep in background thread so main loop isn't blocked
        def _mock_stop():
            print(f"[MOCK] Stop train signal asserted for {duration_seconds}s")
            time.sleep(duration_seconds)
            print("[MOCK] Stop signal released")

        threading.Thread(target=_mock_stop, daemon=True).start()
        return True

# ------------------
# Alerts: Twilio + webhook
# ------------------
def send_twilio_sms(body):
    cfg = CONFIG["twilio"]
    if not cfg.get("enabled"):
        print("Twilio disabled in config; skipping SMS")
        return False
    if not TWILIO_AVAILABLE:
        print("Twilio package not installed; install `twilio` to enable SMS")
        return False
    try:
        client = TwilioClient(cfg.get("account_sid"), cfg.get("auth_token"))
        msg = client.messages.create(body=body, from_=cfg.get("from_number"), to=cfg.get("to_number"))
        print("Twilio SMS sent, sid:", msg.sid)
        return True
    except Exception as e:
        print("Failed to send Twilio SMS:", e)
        return False


def send_webhook(payload, image_path=None):
    url = CONFIG.get("webhook_url")
    if not url:
        return False
    try:
        files = None
        if image_path and os.path.exists(image_path):
            files = {"image": open(image_path, "rb")}
            data = {"payload": json.dumps(payload)}
            r = requests.post(url, data=data, files=files, timeout=8)
        else:
            r = requests.post(url, json=payload, timeout=8)
        r.raise_for_status()
        print("Webhook sent, status", r.status_code)
        return True
    except Exception as e:
        print("Webhook send failed:", e)
        return False

# ------------------
# Simple tracker for debounce
# ------------------

def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

class TrackedObject:
    def __init__(self, bbox, label, confidence, track_id):
        self.bbox = bbox
        self.label = label
        self.confidence = confidence
        self.id = track_id
        self.hits = 1
        self.last_seen = 0
        self.alerted = False
        self.last_alert_time = 0

class SimpleTracker:
    def __init__(self, iou_threshold=0.4, max_age=5, persistence_frames=3, cooldown_seconds=30):
        self.next_id = 1
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.persistence_frames = persistence_frames
        self.cooldown_seconds = cooldown_seconds

    def update(self, detections, frame_idx):
        # detections: list of dicts with keys: bbox, label, confidence
        matched = set()
        # matching: greedy by best iou
        for det in detections:
            best_track = None
            best_iou = 0
            for tr in self.tracks:
                val = iou(det["bbox"], tr.bbox)
                if val > best_iou:
                    best_iou = val
                    best_track = tr
            if best_iou > self.iou_threshold and best_track is not None:
                # update track
                best_track.bbox = det["bbox"]
                best_track.label = det["label"]
                best_track.confidence = det["confidence"]
                best_track.hits += 1
                best_track.last_seen = frame_idx
                matched.add(best_track.id)
            else:
                # create new track
                tr = TrackedObject(det["bbox"], det["label"], det["confidence"], self.next_id)
                tr.last_seen = frame_idx
                self.next_id += 1
                self.tracks.append(tr)
                matched.add(tr.id)

        # age and remove old tracks
        alive = []
        for tr in self.tracks:
            if frame_idx - tr.last_seen <= self.max_age:
                alive.append(tr)
        self.tracks = alive

        # Determine which tracks should trigger alerts
        alerts = []
        for tr in self.tracks:
            if tr.hits >= self.persistence_frames and not tr.alerted:
                # check cooldown
                now_ts = time.time()
                if now_ts - tr.last_alert_time >= self.cooldown_seconds:
                    alerts.append(tr)
                    tr.alerted = True
                    tr.last_alert_time = now_ts
        return alerts

# ------------------
# Main detection loop
# ------------------

def make_snapshot_and_save(frame):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(CONFIG["output_dir"], f"snapshot_{ts}.jpg")
    try:
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, format="JPEG")
        return path
    except Exception as e:
        print("Failed to save snapshot:", e)
        return None


def alert_procedure(tracked_obj, frame):
    # Build payload
    ts = datetime.utcnow().isoformat() + "Z"
    loc = get_location()
    bbox = tracked_obj.bbox
    payload = {
        "timestamp_utc": ts,
        "label": tracked_obj.label,
        "threat_type": CONFIG["threat_classes"].get(tracked_obj.label, "obstacle"),
        "confidence": tracked_obj.confidence,
        "bbox": bbox,
        "location": loc,
        "note": "Debounced persistent detection on railway track",
    }

    # Snapshot
    snapshot_path = make_snapshot_and_save(frame)
    if snapshot_path:
        payload["snapshot_path"] = snapshot_path

    # Save to DB
    dbrec = {
        "timestamp_utc": ts,
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
        "method": loc.get("method"),
        "label": tracked_obj.label,
        "threat_type": CONFIG["threat_classes"].get(tracked_obj.label, "obstacle"),
        "confidence": tracked_obj.confidence,
        "bbox": bbox,
        "snapshot_path": snapshot_path,
    }
    save_alert_db(dbrec)

    # Send notifications
    sms_body = f"Rail Alert: {tracked_obj.label} detected at {ts}, loc: {loc.get('latitude')},{loc.get('longitude')}, confidence: {tracked_obj.confidence:.2f}"
    threading.Thread(target=send_twilio_sms, args=(sms_body,), daemon=True).start()
    threading.Thread(target=send_webhook, args=(payload, snapshot_path), daemon=True).start()

    # Stop train
    stop_train()

    print("ALERT triggered:", json.dumps(payload, indent=2))


def main():
    print("Loading YOLO model:", CONFIG["yolo_model"])
    model = YOLO(CONFIG["yolo_model"])

    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {CONFIG['video_source']}")

    tracker = SimpleTracker(
        iou_threshold=0.3,
        max_age=10,
        persistence_frames=CONFIG["alert_persistence_frames"],
        cooldown_seconds=CONFIG["alert_cooldown_seconds"],
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or frame fetch failed")
                break
            frame_idx += 1

            # Run detection on frame
            results = model(frame, conf=CONFIG["conf_threshold"], imgsz=640)
            res = results[0]
            boxes = getattr(res, "boxes", None)

            detections = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
                    cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                    cls_name = model.names[cls_id]
                    if conf < CONFIG["conf_threshold"]:
                        continue
                    if cls_name not in CONFIG["threat_classes"]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({"bbox": [x1, y1, x2, y2], "label": cls_name, "confidence": conf})

            # Update tracker and get alerts
            alerts = tracker.update(detections, frame_idx)

            # Annotate frame & handle alerts
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['confidence']:.2f}"
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for tr in alerts:
                # trigger alert procedure (non-blocking)
                threading.Thread(target=alert_procedure, args=(tr, frame.copy()), daemon=True).start()

            if CONFIG.get("show_preview"):
                cv2.imshow("Railtech Detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
