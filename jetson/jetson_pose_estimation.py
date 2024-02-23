from ultralytics import YOLO
import json

# 모델 초기화
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

# 포즈 추정 실행
results = model(source=0, device=0, conf=0.35, save=True, show=True)

# 포즈 추정 결과를 JSON 형식으로 변환 및 저장
# results 객체에서 필요한 데이터 추출
poses = []
if hasattr(results, 'pred') and len(results.pred) > 0:
    for i, det in enumerate(results.pred[0]):  # 첫 번째 이미지/프레임에 대한 검출 결과 순회
        if len(det) == 0:
            continue
        # 바운딩 박스, 신뢰도, 클래스 등의 정보 추출
        bbox = det[:, :4].tolist()  # 바운딩 박스 좌표
        conf = det[:, 4].tolist()   # 신뢰도 점수
        cls = det[:, 5].tolist()    # 클래스 ID
        poses.append({
            "bbox": bbox,
            "conf": conf,
            "class": cls
        })

# JSON 파일로 저장
json_filename = "pose_estimation_results.json"
with open(json_filename, "w") as f:
    json.dump(poses, f, indent=4)

print(f"Results saved to {json_filename}")