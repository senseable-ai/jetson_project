from ultralytics import YOLO
model = YOLO("yolov8n-pose-sailab-perfect-fps10.engine")

results = model(source = 0, device = 0, conf = 0.35, save = True, show = True, save_txt = True)

