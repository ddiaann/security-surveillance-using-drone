import cv2
import numpy as np
import time
from djitellopy import Tello
import os  # <-- NEW: Imported to handle the auto-increment file saving

# --- CONFIGURATION ---
prototxt = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.prototxt"
modelpath = r"C:\Users\Asus\Downloads\FYP1\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, modelpath)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Flight Control Variables
rifX, rifY = 960/2, 720/2
Kp_X, Kp_Y, Kp_dist = 0.2, 0.2, 0.05
target_bbox_height = 650

# --- INITIALIZE DRONE ---
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")
drone.streamon()

# --- NEW CODE: AUTO-INCREMENT VIDEO WRITER SETUP ---
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
base_name = "drone_tracking_output_"
ext = ".mp4"
file_index = 1

# Check if the file exists, if it does, increase the number by 1
while os.path.exists(f"{base_name}{file_index}{ext}"):
    file_index += 1

output_filename = f"{base_name}{file_index}{ext}"
# Note: The dimensions (960, 720) must exactly match the resized frame below
out = cv2.VideoWriter(output_filename, fourcc, fps, (960, 720))
print(f"Saving video to: {output_filename}")
# ---------------------------------------------------

print("Taking off in 3 seconds...")
time.sleep(3)
drone.takeoff()
drone.move_up(30) # Get to eye-level

print("Tracking Started. Press 'q' to land and quit.")

while True:
    frame = drone.get_frame_read().frame
    if frame is None: continue
    
    # --- NEW CODE: COLOR FIX ---
    # Convert color from BGR to RGB to fix the video stream color mapping
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ---------------------------

    # Resize frame (Matches the VideoWriter dimensions)
    frame = cv2.resize(frame, (960, 720))
    h, w, _ = frame.shape

    # Detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()

    tracked = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person" and confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw Box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, f"Tracking | Conf: {confidence*100:.0f}%", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # --- PI CONTROL LOGIC ---
            centroX = (startX + endX) / 2
            centroY = (2 * startY + endY) / 3
            bbox_h = endY - startY

            error_X = -(rifX - centroX)
            error_Y = rifY - centroY
            
            move_yaw = int(np.clip(Kp_X * error_X, -30, 30))
            move_updown = int(np.clip(Kp_Y * error_Y, -20, 20))
            move_fb = int(np.clip(Kp_dist * (target_bbox_height - bbox_h), -30, 30))
            
            # Send commands to Tello
            drone.send_rc_control(0, move_fb, move_updown, move_yaw)
            tracked = True
            break # Only track the first person found in this basic script

    if not tracked:
        drone.send_rc_control(0, 0, 0, 0) # Hover in place if no one is seen

    # --- WRITE FRAME TO MP4 ---
    out.write(frame)
    # ------------------------------------

    cv2.imshow("Subsystem 2: Drone Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
out.release()  # Close the video file properly
drone.land()
drone.streamoff()
cv2.destroyAllWindows()
drone.end()
print(f"Process Complete. Video successfully saved as {output_filename}.")