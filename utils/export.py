from ultralytics import YOLO
YOLO('/home/jasperxzy/racing/racing_segmentation/runs/train/yolo11n-seg/weights/best.pt').export(imgsz=640, format='onnx', simplify=False, opset=11)