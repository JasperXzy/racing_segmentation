from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="/home/jasperxzy/dataset/racing_seg/dataset.yaml", 
                      epochs=150, 
                      imgsz=640, 
                      workers=16,
                      batch=0.95, 
                      device='0',
                      project='./runs/train',
                      name='yolo11n-seg',
                      cache=True,
                      )
