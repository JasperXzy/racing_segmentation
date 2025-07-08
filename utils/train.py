from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="/home/jasperxzy/racing/racing_segmentation/dataset/dataset.yaml", 
                      epochs=100, 
                      imgsz=1080, 
                      batch=8, 
                      device='0',
                      project='./runs/train',
                      name='yolo11n-seg',
                      cache=True,
                      )
