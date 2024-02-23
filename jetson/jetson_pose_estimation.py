from ultralytics import YOLO
import json

# 모델 초기화
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# 포즈 추정 실행
results = model(source=0, device=0, conf=0.35, save=True, show=True)

# 결과를 JSON 형식으로 변환 및 저장
json_results = results.tojson()

# JSON 파일로 저장
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as f:
    f.write(json_results)

print(f"Results saved to {json_filename}")