import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
import time
from djitellopy import Tello
import os 

# ============================
# DEVICE & CONFIGURATION
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

prototxt = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.prototxt"
modelpath = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Load MobileNetSSD
net = cv2.dnn.readNetFromCaffe(prototxt, modelpath)

# Load OpenCLIP
print("Loading OpenCLIP Model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()

def extract_feature(crop_img, is_already_rgb=False):
    # Safeguard to prevent "double-swapping" colors if the frame was already converted
    if is_already_rgb:
        crop_rgb = crop_img
    else:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
    img = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat

# ============================
# SHARED MEMORY (THE "BRAIN")
# ============================
known_targets = {}  # Stores {ID: feature_tensor}
id_origins = {}     # Stores which camera first saw the person {ID: "Static" or "Drone"}
next_id = 1         # Counter for new people

def identify_and_draw(frame, window_name, is_rgb_input=False):
    global next_id
    h, w, _ = frame.shape
    
    # Track used IDs per frame to prevent Identity Cloning
    used_ids_this_frame = []
    
    # Detect humans
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            crop = frame[startY:endY, startX:endX]

            if crop.size == 0: continue

            # Extract visual signature (passes the RGB flag to prevent color bugs)
            feat = extract_feature(crop, is_already_rgb=is_rgb_input)

            # Compare against our memory
            best_id = None
            best_sim = 0
            for tid, tfeat in known_targets.items():
                if tid in used_ids_this_frame:
                    continue  

                sim = torch.cosine_similarity(feat, tfeat).item()
                if sim > best_sim:
                    best_sim = sim
                    best_id = tid

            # -----------------------------------------------------
            # THE RE-ID LOGIC (Threshold set to 0.78 for Handoff)
            # -----------------------------------------------------
            if best_sim > 0.78 and best_id is not None:
                display_id = best_id
                display_sim = best_sim * 100
                color = (0, 255, 0) # Green for known match
                
                used_ids_this_frame.append(display_id)
                
                # Update memory to learn the new angle
                known_targets[best_id] = (known_targets[best_id] + feat) / 2
                known_targets[best_id] /= known_targets[best_id].norm(dim=-1, keepdim=True)

                # --- CROSS-CAMERA HANDOFF ALERT ---
                if id_origins[display_id] != window_name:
                    cv2.putText(frame, "CROSS-CAMERA HANDOFF!", (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    color = (0, 255, 255) # Turn box yellow to highlight the handoff

            else:
                # This is a new person we haven't seen before
                display_id = next_id
                display_sim = 100.0 
                color = (0, 165, 255) # Orange for new person
                print(f"[{window_name}] New Person Detected! Assigned ID: Person {display_id}")
                
                used_ids_this_frame.append(display_id)
                known_targets[display_id] = feat
                id_origins[display_id] = window_name # Remember who saw them first
                next_id += 1

            # Draw bounding box and ID
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label = f"Person {display_id} | {display_sim:.1f}%"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ============================
# HARDWARE INITIALIZATION
# ============================
print("Opening Static Camera...")
cap = cv2.VideoCapture(1) # Change to 0 if your webcam doesn't open
assert cap.isOpened(), "Static camera not found!"

print("Connecting to Drone...")
drone = Tello()
drone.connect()
print(f"Drone Battery: {drone.get_battery()}%")
drone.streamon()
time.sleep(2) 

# ============================
# AUTO-INCREMENT VIDEO WRITER 
# ============================
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
base_name = "dual_camera_reid_output_"
ext = ".mp4"
file_index = 1

while os.path.exists(f"{base_name}{file_index}{ext}"):
    file_index += 1

output_filename = f"{base_name}{file_index}{ext}"
out = cv2.VideoWriter(output_filename, fourcc, fps, (1280, 480))
print(f"Saving dual-feed video to: {output_filename}")

print("Dual-Camera Re-ID System Active. Press 'q' to quit.")

# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame_s = cap.read()
    if not ret: continue
    
    frame_d_raw = drone.get_frame_read().frame
    if frame_d_raw is None: continue

    # --- COLOR FORMAT CONVERSION ---
    # Convert Tello's raw stream as requested
    frame_d_rgb = cv2.cvtColor(frame_d_raw, cv2.COLOR_BGR2RGB)

    # Process both frames through our shared memory brain
    # We tell the function that the drone frame is already RGB, so it doesn't double-swap!
    frame_s_processed = identify_and_draw(frame_s.copy(), "Static", is_rgb_input=False)
    frame_d_processed = identify_and_draw(frame_d_rgb.copy(), "Drone", is_rgb_input=True)

    # Resize them so they fit next to each other perfectly
    frame_s_resized = cv2.resize(frame_s_processed, (640, 480))
    frame_d_resized = cv2.resize(frame_d_processed, (640, 480))

    cv2.putText(frame_s_resized, "STATIC CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_d_resized, "DRONE CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stitch them together (1280 x 480 output)
    combined_feed = np.hstack((frame_s_resized, frame_d_resized))

    # Save to MP4
    out.write(combined_feed)

    # Show the result
    cv2.imshow("Subsystem 3: Dual Camera Re-ID Test", combined_feed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================
# CLEANUP
# ============================
out.release() 
cap.release()
drone.streamoff()
cv2.destroyAllWindows()
drone.end()
print(f"System Offline. Video successfully saved as {output_filename}.")