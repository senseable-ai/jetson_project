from ultralytics import YOLO
import cv2
import json
import os

def run_yolo_webcam_detection():
    # Load a pretrained YOLO model
    model = YOLO('yolov8n-pose.pt')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    try:
        while True:
            # Read frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the current frame
            results = model(frame, show=True)

            # Prepare data for JSON
            # 파일이 존재하면 기존 데이터 로드, 없으면 빈 리스트 초기화
            if os.path.exists('detections.json'):
                with open('detections.json', 'r') as infile:
                    data_to_save = json.load(infile)
            else:
                data_to_save = []

            for r in results:
                keypoints = r.keypoints.data.tolist()  # Convert keypoints to list for JSON serialization
                data_to_save.append({
                    "keypoints": keypoints,
                    "confidence": r.keypoints.conf.tolist(),  # Confidence scores included
                })

            # 변경된 전체 데이터를 파일에 다시 저장
            with open('detections.json', 'w') as outfile:
                json.dump(data_to_save, outfile, indent=4, sort_keys=True)

            # Display the frame (optional)
            cv2.imshow('YOLOv8 Real-Time Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the webcam and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_webcam_detection()
