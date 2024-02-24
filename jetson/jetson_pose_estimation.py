from ultralytics import YOLO
import cv2
import time
import json

# Initialize the model
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# Open video source; 0 for webcam
cap = cv2.VideoCapture(0)

prev_frame_time = 0
json_results = None  # Initialize variable to hold JSON results

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = model(frame)

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"

    # Add FPS text to frame
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

    # Render pose estimation on the frame
    frame = results.render()[0]

    # Show the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Prepare results for JSON conversion
    json_results = results.pandas().xyxy[0].to_json(orient="records", indent=4)  # Modified for readability

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Save the last frame's results to JSON with better readability
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as f:
    f.write(json_results)

print(f"Results saved to {json_filename}")
