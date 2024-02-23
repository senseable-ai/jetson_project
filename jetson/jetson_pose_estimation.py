from ultralytics import YOLO
import cv2
import time
import json

# Initialize the model
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# Open video source; 0 for webcam, or 'path/to/video' for video files
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    # Note: Adjust this part if model inference on a frame is different
    results = model(frame)
    json_results = results.tojson()

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the results to a JSON file
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as f:
    f.write(json_results)

print(f"Results saved to {json_filename}")