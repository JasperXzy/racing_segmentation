from ultralytics import YOLO
import cv2
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation Inference')
    parser.add_argument('--source', type=str, help='Path to video/image/directory')
    parser.add_argument('--model', type=str, help='Path to model weights')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference size (pixels) (default: 640)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--display-size', type=int, nargs=2, default=[1920, 1080],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Display window size (default: 1920 1080)')
    args = parser.parse_args()

    # Initialize YOLO model
    model = YOLO(args.model)
    display_w, display_h = args.display_size  # Get display dimensions

    # Determine input source type
    source = args.source
    is_video = False
    is_image = False
    is_dir = False
    
    # Check source type
    if os.path.isdir(source):
        is_dir = True
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        # Collect image files from directory
        image_files = [os.path.join(source, f) for f in os.listdir(source) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        if not image_files:
            print(f"Error: No supported images found in directory {image_extensions}")
            return
    elif os.path.isfile(source):
        # Check file extension
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video = True
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            is_image = True
        else:
            print(f"Error: Unsupported file format {ext}")
            return
    elif source.isdigit():
        is_video = True  # Treat integer as webcam ID
    else:
        print("Error: Invalid input source")
        return

    # Process video/webcam input
    if is_video:
        # Initialize video capture (webcam if source is integer)
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Perform YOLO inference on frame
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            
            # Visualize results if available
            if results:
                annotated_frame = results[0].plot()  # Get annotated frame
                # Resize to display dimensions
                annotated_frame = cv2.resize(annotated_frame, (display_w, display_h))
                cv2.imshow("YOLO11 Segment", annotated_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    # Process single image
    elif is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: Could not read image {source}")
            return

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        if results:
            annotated_frame = results[0].plot()
            annotated_frame = cv2.resize(annotated_frame, (display_w, display_h))
            cv2.imshow("YOLO11 Segment", annotated_frame)
            cv2.waitKey(0)  # Wait for any key press
            cv2.destroyAllWindows()

    # Process directory of images
    elif is_dir:
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Skipping unreadable image {img_path}")
                continue

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            if results:
                annotated_frame = results[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (display_w, display_h))
                cv2.imshow("YOLO11 Segment", annotated_frame)
                
                key = cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
