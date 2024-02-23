from ultralytics import YOLO
import json

# 모델 초기화
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# 포즈 추정 실행
results = model(source=0, device=0, conf=0.35, save=True, show=True, save_txt=True)

# 결과를 pandas 데이터프레임으로 변환
df = results.pandas().xyxy[0]  # 첫 번째 이미지의 결과

# 데이터프레임을 딕셔너리로 변환 (예: 클래스, 확률, 바운딩 박스 좌표)
results_dict = df.to_dict("records")

# 결과를 JSON 파일로 저장
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as outfile:
    json.dump(results_dict, outfile, indent=4)

print(f"Results saved to {json_filename}")
