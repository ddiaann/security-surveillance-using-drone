import cv2
import numpy as np
import torch
import pandas as pd
import time
import os
from PIL import Image
import open_clip
from djitellopy import Tello
from datetime import datetime

# ============================
# 1. CONFIGURATION & SETUP
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
prototxt = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.prototxt"
modelpath = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

GROUND_TRUTH_ID = 1 
rifX, rifY = 960/2, 720/2
Kp_X, Kp_Y, Kp_dist = 0.25, 0.25, 0.05
target_bbox_height = 500

# ============================
# 2. LOAD MODELS
# ============================
net = cv2.dnn.readNetFromCaffe(prototxt, modelpath)
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()

def extract_feature(crop_img, is_rgb=False):
    if not is_rgb:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = preprocess(Image.fromarray(crop_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat

# ============================
# 3. SHARED MEMORY & LOGGING
# ============================
known_targets = {}
id_origins = {}
next_id = 1
log_data = []

# File Naming
file_idx = 1
while os.path.exists(f"experiment_log_{file_idx}.xlsx"): file_idx += 1
excel_path = f"experiment_log_{file_idx}.xlsx"
video_path = f"integrated_system_output_{file_idx}.mp4"

# ============================
# 4. HARDWARE INITIALIZATION
# ============================
cap = cv2.VideoCapture(1) 
drone = Tello()
drone.connect()
drone.streamon()
print(f"Battery: {drone.get_battery()}%")

# Video Writer - Using 'mp4v' for high compatibility
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 480))

# ============================
# 5. STATE VARIABLES
# ============================
drone_is_airborne = False
person_seen_by_static = False
takeoff_ready = False # This becomes True once person leaves FOV

def process_frame(frame, cam_label, is_rgb=False):
    global next_id, person_seen_by_static
    h, w, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()
    
    best_frame_data = None
    target_in_this_frame = False
    used_ids_this_frame = []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.6 and CLASSES[int(detections[0, 0, i, 1])] == "person":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size == 0: continue

            feat = extract_feature(crop, is_rgb=is_rgb)
            rankings = []
            for tid, tfeat in known_targets.items():
                if tid in used_ids_this_frame: continue
                sim = torch.cosine_similarity(feat, tfeat).item()
                rankings.append({"id": tid, "sim": sim})
            rankings = sorted(rankings, key=lambda x: x['sim'], reverse=True)
            
            best_id, best_sim = (rankings[0]['id'], rankings[0]['sim']) if rankings else (None, 0)

            if best_sim > 0.78:
                display_id = best_id
                color = (0, 255, 0)
                known_targets[display_id] = (known_targets[display_id] + feat) / 2
                known_targets[display_id] /= known_targets[display_id].norm(dim=-1, keepdim=True)
                if id_origins[display_id] != cam_label:
                    color = (0, 255, 255)
                    cv2.putText(frame, "HANDOVER", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                display_id = next_id
                known_targets[display_id] = feat
                id_origins[display_id] = cam_label
                next_id += 1
                color = (0, 165, 255)
                best_sim = 1.0

            if display_id == GROUND_TRUTH_ID: target_in_this_frame = True
            used_ids_this_frame.append(display_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{display_id} | {best_sim*100:.0f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if best_frame_data is None:
                best_frame_data = {"box": (x1, y1, x2, y2), "conf": conf, "rankings": rankings, "final_id": display_id}

    if cam_label == "Static":
        if target_in_this_frame: person_seen_by_static = True
        return frame, target_in_this_frame
    
    return frame, best_frame_data

# ============================
# 6. MAIN LOOP
# ============================
print("System Active. Monitoring Static Camera...")
try:
    while True:
        start_time = time.time()
        ret, frame_s = cap.read()
        frame_d_raw = drone.get_frame_read().frame
        if not ret or frame_d_raw is None: continue

        # Standardize colors
        frame_d_rgb = cv2.cvtColor(frame_d_raw, cv2.COLOR_BGR2RGB)
        
        # Process Feeds
        proc_s, target_present_static = process_frame(frame_s.copy(), "Static", False)
        proc_d, data_d = process_frame(frame_d_rgb.copy(), "Drone", True)

        # TAKEOFF LOGIC: Wait until person is seen THEN leaves FOV
        if not drone_is_airborne:
            if person_seen_by_static and not target_present_static:
                print("Target left Static Camera FOV. Drone taking off!")
                drone.takeoff()
                drone.move_up(50)
                drone_is_airborne = True
            else:
                cv2.putText(proc_s, "WAITING FOR TARGET TO LEAVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Drone Control
        if drone_is_airborne:
            if data_d:
                x1, y1, x2, y2 = data_d["box"]
                cx, cy = (x1 + x2) / 2, (2 * y1 + y2) / 3
                bh = y2 - y1
                m_yaw = int(np.clip(Kp_X * (cx - rifX), -30, 30))
                m_ud = int(np.clip(Kp_Y * (rifY - cy), -25, 25))
                m_fb = int(np.clip(Kp_dist * (target_bbox_height - bh), -20, 20))
                drone.send_rc_control(0, m_fb, m_ud, m_yaw)
            else:
                drone.send_rc_control(0, 0, 0, 0)

        # Log Data
        fps = 1.0 / (time.time() - start_time)
        if data_d and drone_is_airborne:
            rankings = data_d["rankings"]
            actual_rank = next((i+1 for i, r in enumerate(rankings) if r['id'] == GROUND_TRUTH_ID), None)
            log_data.append({
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "drone_conf": data_d["conf"], "predicted_id": data_d["final_id"],
                "rank_of_truth": actual_rank, "fps": fps
            })

        # Stitch and Write
        combined = np.hstack((cv2.resize(proc_s, (640, 480)), cv2.resize(proc_d, (640, 480))))
        out.write(combined)
        cv2.imshow("Handoff System", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Landing drone...")
            break

finally:
    # Shutdown sequence
    if drone_is_airborne:
        drone.send_rc_control(0,0,0,0)
        drone.land()
    
    df = pd.DataFrame(log_data)
    df.to_excel(excel_path, index=False)
    
    cap.release()
    out.release() # CRITICAL: This saves the mp4 file
    drone.streamoff()
    cv2.destroyAllWindows()
    drone.end()
    print(f"File Saved: {video_path}")