from ultralytics import YOLO

# 모델 초기화
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# 포즈 추정 실행
results = model(source=0, device=0, conf=0.35, save=True, show=True)

# 포즈 추정 결과를 JSON 형식으로 변환 및 저장 (수정된 부분)
# `tojson()` 메소드 사용
json_results = results.tojson()  # 결과를 직접 JSON 문자열로 변환

# JSON 파일로 저장 (수정된 부분)
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as f:
    f.write(json_results)  # 직접 변환된 JSON 문자열을 파일에 쓰기

print(f"Results saved to {json_filename}")
