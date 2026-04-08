import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
import os

# ============================
# 1. CONFIGURATION
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

prototxt = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.prototxt"
modelpath = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Load MobileNetSSD (Detection)
net = cv2.dnn.readNetFromCaffe(prototxt, modelpath)

# Load OpenCLIP (Memory/Re-ID)
print("Loading OpenCLIP Model for Identity Tracking...")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()

def extract_feature(crop_img):
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat

# ============================
# 2. SHARED MEMORY SETUP
# ============================
known_targets = {}  # Dictionary to store {ID: feature_tensor}
next_id = 1         # Counter for new people

# ============================
# 3. CAMERA & VIDEO WRITER
# ============================
cap = cv2.VideoCapture(1) 

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
base_name = "static_tracking_output_"
ext = ".mp4"
file_index = 1

while os.path.exists(f"{base_name}{file_index}{ext}"):
    file_index += 1

output_filename = f"{base_name}{file_index}{ext}"
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
print(f"Saving video to: {output_filename}")

print("Starting Static Camera Detection & Tracking. Press 'q' to quit.")

# ============================
# 4. MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    # Detect Humans
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()

    # --- NEW: Track which IDs have been claimed in this exact frame ---
    used_ids_this_frame = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            crop = frame[startY:endY, startX:endX]

            if crop.size == 0: continue

            feat = extract_feature(crop)

            best_id = None
            best_sim = 0
            
            # Compare against memory
            for tid, tfeat in known_targets.items():
                # NEW LOGIC: If this ID was already assigned to someone else in this frame, skip it!
                if tid in used_ids_this_frame:
                    continue

                sim = torch.cosine_similarity(feat, tfeat).item()
                if sim > best_sim:
                    best_sim = sim
                    best_id = tid

            # --- NEW: Tightened Threshold to 0.82 to prevent identity merging ---
            if best_sim > 0.82 and best_id is not None:
                display_id = best_id
                color = (0, 255, 0) # Green for recognized
                
                # Lock this ID so no other box in this frame can steal it
                used_ids_this_frame.append(display_id)
                
                # Smoothly update memory
                known_targets[best_id] = (known_targets[best_id] + feat) / 2
                known_targets[best_id] /= known_targets[best_id].norm(dim=-1, keepdim=True)
            else:
                display_id = next_id
                color = (0, 165, 255) # Orange for new
                
                # Lock the new ID
                used_ids_this_frame.append(display_id)
                
                print(f"New Person Detected! Assigned ID: Person {display_id} (Sim was {best_sim:.2f})")
                
                known_targets[display_id] = feat
                next_id += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label = f"Person {display_id} | {confidence * 100:.0f}%"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Subsystem 1: Identity Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================
# 5. CLEANUP
# ============================
out.release()
cap.release()
cv2.destroyAllWindows()
print(f"Process Complete. Video successfully saved as {output_filename}.")