import cv2
import numpy as np
import os
import glob

def replace_qrcodes_in_image(input_image_path, replacements_folder_path, output_image_path):
    """
    检测图像中的二维码，并用文件夹中的其他二维码进行替换。

    参数:
    input_image_path (str): 带有二维码的原始图片路径。
    replacements_folder_path (str): 包含替换用二维码图片的文件夹路径。
    output_image_path (str): 处理后图片的保存路径。
    """
    # 1. 加载替换用的二维码图片
    replacement_image_paths = sorted(glob.glob(os.path.join(replacements_folder_path, '*.[pP][nN][gG]')) + \
                                     glob.glob(os.path.join(replacements_folder_path, '*.[jJ][pP][gG]')))
    
    if not replacement_image_paths:
        print(f"错误：在文件夹 '{replacements_folder_path}' 中找不到任何替换用的二维码图片（.png 或 .jpg）。")
        return

    replacement_images = [cv2.imread(p) for p in replacement_image_paths]
    print(f"找到了 {len(replacement_images)} 个替换用的二维码。")

    # 2. 加载原始图片
    main_image = cv2.imread(input_image_path)
    if main_image is None:
        print(f"错误：无法加载原始图片 '{input_image_path}'。")
        return
    
    # 复制一份用于修改
    output_image = main_image.copy()

    # 3. 初始化二维码检测器并检测
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(main_image)

    if retval:
        print(f"在原图中检测到 {len(points)} 个二维码。")
        
        # 4. 遍历并替换每一个检测到的二维码
        for i, pts in enumerate(points):
            # 获取当前用于替换的二维码（如果替换图片不够，则循环使用）
            replacement_qr = replacement_images[i % len(replacement_images)]
            h, w = replacement_qr.shape[:2]

            # a. 定义新二维码的源角点（标准矩形）
            # 角点顺序必须与检测器返回的顺序对应: 左上, 右上, 右下, 左下
            src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

            # b. 目标角点就是检测器返回的角点
            dst_pts = np.array(pts, dtype=np.float32)

            # c. 计算透视变换矩阵 (Homography)
            homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)

            # d. 将新二维码进行透视变换，使其扭曲成目标形状
            # dsize 是目标图像的大小，即原始大图的大小
            warped_qr = cv2.warpPerspective(replacement_qr, homography_matrix, (main_image.shape[1], main_image.shape[0]))

            # e. 创建掩膜 (Mask) 以便无缝粘贴
            # 创建一个全黑的掩膜
            mask = np.zeros_like(main_image, dtype=np.uint8) 
            # 在掩膜上将目标二维码区域填充为白色
            cv2.fillConvexPoly(mask, dst_pts.astype(int), (255, 255, 255))
            # 掩膜取反，用于在原图上“挖洞”
            inverse_mask = cv2.bitwise_not(mask)
            
            # f. 组合图像
            # 在原图上挖出旧二维码的洞（区域变黑）
            temp_img = cv2.bitwise_and(output_image, inverse_mask)
            # 将扭曲后的新二维码与挖洞后的原图相加，完成替换
            output_image = cv2.add(temp_img, warped_qr)
            
            print(f"已替换第 {i+1} 个二维码。")

    else:
        print("在原图中未检测到任何二维码。")
        
    # 5. 保存结果
    cv2.imwrite(output_image_path, output_image)
    print(f"处理完成，结果已保存到 '{output_image_path}'。")

    # (可选) 显示结果以供预览
    # cv2.imshow("Original Image", main_image)
    # cv2.imshow("Replaced QR Code Image", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # --- 配置区域 ---
    # 原始图片路径
    INPUT_IMAGE = 'source_image.jpg'
    # 替换用二维码所在的文件夹路径
    REPLACEMENTS_FOLDER = 'replacement_qrcodes'
    # 输出图片路径
    OUTPUT_IMAGE = 'result_image.jpg'
    # --- 配置结束 ---

    # 确保替换文件夹存在
    if not os.path.exists(REPLACEMENTS_FOLDER):
        os.makedirs(REPLACEMENTS_FOLDER)
        print(f"创建了文件夹 '{REPLACEMENTS_FOLDER}'。请向其中添加替换用的二维码图片。")
    else:
        replace_qrcodes_in_image(INPUT_IMAGE, REPLACEMENTS_FOLDER, OUTPUT_IMAGE)
