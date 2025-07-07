from ultralytics import YOLO
import cv2

model = YOLO('/home/jasperxzy/racing/racing_segmentation/runs/train/yolo11n-seg/weights/best.pt')  
video_path = '/home/jasperxzy/racing/racing_segmentation/dataset/raw_video/IMG_3023.MOV'                     

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, stream=True, imgsz=640, conf=0.3)

    display_w, display_h = 1920, 1080
 
    for result in results:
        im = result.plot()  

        im = cv2.resize(im, (display_w, display_h))
 
        cv2.imshow("YOLO11 Segment", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
cap.release()
cv2.destroyAllWindows()