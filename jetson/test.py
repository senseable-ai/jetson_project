from ultralytics import YOLO
import json

# 모델 초기화
model = YOLO('yolov8n-pose.pt')

# 모델 실행
results = model(source=0, device=0, save=True, save_txt=True, show=True)

# 키포인트 식별자와 이름 매핑
keypoint_names = {
    0: "nose",
    1: "right_eye",
    2: "left_eye",
    3: "right_ear",
    4: "left_ear",
    5: "right_shoulder",
    6: "left_shoulder",
    7: "right_elbow",
    8: "left_elbow",
    9: "right_wrist",
    10: "left_wrist",
    11: "right_hip",
    12: "left_hip",
    13: "right_knee",
    14: "left_knee",
    15: "right_ankle",
    16: "left_ankle"
}

# 결과를 JSON 형식으로 변환
def results_to_json(results):
    json_data = []
    for result in results.xyxy[0]:  # results.xyxy[0]는 검출된 객체들의 바운딩 박스와 클래스 정보를 담고 있음
        # 각 객체에 대한 정보 추출
        class_id = int(result[-1])
        confidence = float(result[4])
        keypoints = [{"part": keypoint_names[i], "x": float(kp[0]), "y": float(kp[1]), "visibility": float(kp[2])} for i, kp in enumerate(result[5:])]
        
        json_data.append({
            "class_id": class_id,
            "confidence": confidence,
            "keypoints": keypoints
        })
    
    return json_data

# 결과를 JSON 파일로 저장
def save_results_to_json_file(results, file_path):
    json_data = results_to_json(results)
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

# 파일 저장
save_results_to_json_file(results, 'detected_poses.json')