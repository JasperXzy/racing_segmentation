import os
import random
from PIL import Image, ImageEnhance

def adjust_hue(img, factor):
    # PIL没有直接调节色调的接口，可以用HSV模式简单变换
    if img.mode != 'RGB':
        img = img.convert('RGB')
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    np_h = h.point(lambda p: (p + int(factor * 255)) % 256)
    hsv = Image.merge('HSV', (np_h, s, v))
    return hsv.convert('RGB')

def augment_image(img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    basename = os.path.basename(img_path)
    
    # 50%概率进行左右镜像
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 随机色调调整（-0.2~0.2之间的变化）
    hue_factor = random.uniform(-0.2, 0.2)
    img = adjust_hue(img, hue_factor)

    # 随机亮度调整（0.7~1.3倍）
    brightness_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 随机对比度调整（0.7~1.3倍）
    contrast_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 随机色度调整（0.7~1.3倍）
    color_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Color(img).enhance(color_factor)

    img.save(save_path)

def augment_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            augment_image(input_path, output_path)
            print(f"Processed {file}")

if __name__ == "__main__":
    input_dir = "/home/jasperxzy/dataset/racing_seg/processed_images"  # 替换为你的图片文件夹路径
    output_dir = "/home/jasperxzy/dataset/racing_seg/images"            # 输出文件夹
    augment_folder(input_dir, output_dir)
