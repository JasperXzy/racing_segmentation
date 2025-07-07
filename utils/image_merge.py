import os
import shutil

main_folder = "/home/jasperxzy/racing/racing_segmentation/dataset/raw_image"

img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in img_exts

def main():
    all_images = []
    for root, dirs, files in os.walk(main_folder):
        if root == main_folder:
            continue
        for f in files:
            if is_image(f):
                all_images.append(os.path.join(root, f))

    total = len(all_images)
    num_width = max(4, len(str(total))) 
    for idx, img_path in enumerate(sorted(all_images), 1):
        new_name = str(idx).zfill(num_width) + '.jpg'
        dst_path = os.path.join(main_folder, new_name)
        shutil.copy2(img_path, dst_path) 

    print(f"Done. {total} images have been merged into '{main_folder}'.")

if __name__ == '__main__':
    main()