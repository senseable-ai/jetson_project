## install
install jetpack 5.1.1

install pytorch / torchvision // 0.12.0 / 0.13.0

trtexec build

install ultralytics

install onnx / onnxruntime / onnxruntime-gpu

git clone ultralytics yolov8

## export
yolo export model=yolov8n-pose.pt format=engine dynamic 
## execute
python3 jetson_pose_estimation.py

