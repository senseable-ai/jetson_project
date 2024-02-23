from ultralytics import YOLO
import cv2
import time

# Initialize the model
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# Open video source; 0 for webcam
cap = cv2.VideoCapture(0)

prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = model(frame)
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"
    
    # Add FPS text to frame
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)
    
    # Show the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Convert results to JSON and save
json_results = results.tojson()
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as f:
    f.write(json_results)

print(f"Results saved to {json_filename}")
