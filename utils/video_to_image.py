import cv2
import os
import glob

def video_to_images(input_folder, output_folder, interval=10):
    os.makedirs(output_folder, exist_ok=True)

    video_files = glob.glob(os.path.join(input_folder, '*.MOV')) + \
                  glob.glob(os.path.join(input_folder, '*.mov')) + \
                  glob.glob(os.path.join(input_folder, '*.mp4')) + \
                  glob.glob(os.path.join(input_folder, '*.avi')) + \
                  glob.glob(os.path.join(input_folder, '*.mkv'))

    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join(output_folder, video_name)
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        save_count = 0
        frame_count = 0
        print(f'[{idx}/{len(video_files)}] Processing: "{video_name}"')
        print(f'Total frames: {total_frames}')

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frame_file = os.path.join(save_dir, f'{frame_count:06d}.jpg')
                cv2.imwrite(frame_file, frame)
                save_count += 1
            frame_count += 1
            print(f'  Progress: {frame_count}/{total_frames} frames...')

        cap.release()
        print(f'Extraction finished for "{video_name}"')
        print(f'Total frames processed: {frame_count}')
        print(f'Total images saved: {save_count}')
        print('-' * 60)

    print("All videos have been processed!")

if __name__ == "__main__":
    input_folder = '/home/jasperxzy/racing/racing_segmentation/dataset/raw_video'
    output_folder = '/home/jasperxzy/racing/racing_segmentation/dataset/raw_image'
    video_to_images(input_folder, output_folder)
    