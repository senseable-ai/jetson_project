from ultralytics import YOLO
import cv2
import json
import os
import threading
import time
from datetime import datetime
import queue  # queue 모듈 임포트 추가

class SaveDataThread(threading.Thread):
    def __init__(self, data_queue):
        threading.Thread.__init__(self)
        self.data_queue = data_queue
        self.daemon = True  # Daemonize thread

    def run(self):
        while True:
            if not self.data_queue.empty():
                data_to_save, filename = self.data_queue.get()
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as outfile:
                    json.dump(data_to_save, outfile, indent=4, sort_keys=True)

def run_yolo_webcam_detection():
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(0)
    data_queue = queue.Queue()  # queue.Queue()로 변경
    SaveDataThread(data_queue).start()

    start_time = time.time()
    data_to_save = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, show=True)
            
            for r in results:
                keypoints = r.keypoints.data.tolist()
                data_to_save.append({
                    "keypoints": keypoints,
                    "confidence": r.keypoints.conf.tolist(),
                })

            # Every 5 minutes, save data to a new file
            if time.time() - start_time > 300:  # 300 seconds = 5 minutes
                filename = f'detections_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
                data_queue.put((data_to_save, filename))
                data_to_save = []  # Reset the data list
                start_time = time.time()  # Reset the timer

            cv2.imshow('YOLOv8 Real-Time Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if data_to_save:  # Save any remaining data
            filename = f'detections_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
            data_queue.put((data_to_save, filename))

if __name__ == "__main__":
    run_yolo_webcam_detection()
