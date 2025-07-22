import os
import uuid

jpg_folder_path = '/home/jasperxzy/dataset/racing_seg/images'  
txt_folder_path = '/home/jasperxzy/dataset/racing_seg/labels'  
# -----------------------------

def rename_files(jpg_dir, txt_dir):
    """
    根据匹配规则重命名jpg和txt文件。

    规则:
    1. 如果 a.jpg 和 a.txt 同时存在，则将它们重命名为 uuid.jpg 和 uuid.txt。
    2. 如果只有 a.jpg 存在，没有 a.txt，则将 a.jpg 重命名为 uuid.jpg。
    """
    print(f"开始处理 JPG 文件夹: {jpg_dir}")
    print(f"对应的 TXT 文件夹: {txt_dir}\n")

    # 检查文件夹是否存在
    if not os.path.isdir(jpg_dir):
        print(f"错误: JPG文件夹路径 '{jpg_dir}' 不存在或不是一个目录。")
        return
    if not os.path.isdir(txt_dir):
        print(f"错误: TXT文件夹路径 '{txt_dir}' 不存在或不是一个目录。")
        return

    # 获取所有JPG文件列表
    try:
        jpg_files = [f for f in os.listdir(jpg_dir) if f.lower().endswith('.jpg')]
    except FileNotFoundError:
        print(f"错误：无法访问JPG文件夹 '{jpg_dir}'")
        return

    if not jpg_files:
        print("JPG文件夹中没有找到任何 .jpg 文件。")
        return

    processed_count = 0
    match_count = 0
    no_match_count = 0

    # 遍历所有JPG文件
    for jpg_filename in jpg_files:
        # 分离文件名和扩展名
        base_name, _ = os.path.splitext(jpg_filename)

        # 构建对应的TXT文件名和完整路径
        txt_filename = base_name + '.txt'
        potential_txt_path = os.path.join(txt_dir, txt_filename)

        # 获取原始JPG文件的完整路径
        original_jpg_path = os.path.join(jpg_dir, jpg_filename)
        
        # 生成一个新的、独一无二的UUID
        new_uuid = str(uuid.uuid4())

        # 检查对应的TXT文件是否存在
        if os.path.isfile(potential_txt_path):
            # --- 情况 1: 存在匹配的TXT文件 ---
            # 定义新的文件名
            new_jpg_filename = f"{new_uuid}.jpg"
            new_txt_filename = f"{new_uuid}.txt"

            # 定义新的完整路径
            new_jpg_path = os.path.join(jpg_dir, new_jpg_filename)
            new_txt_path = os.path.join(txt_dir, new_txt_filename)

            # 重命名文件
            try:
                os.rename(original_jpg_path, new_jpg_path)
                os.rename(potential_txt_path, new_txt_path)
                print(f"匹配成功: [{jpg_filename}, {txt_filename}] -> 重命名为 [{new_jpg_filename}, {new_txt_filename}]")
                match_count += 1
            except OSError as e:
                print(f"重命名文件时发生错误: {e}")

        else:
            # --- 情况 2: 不存在匹配的TXT文件 ---
            # 定义新的JPG文件名
            new_jpg_filename = f"{new_uuid}.jpg"

            # 定义新的JPG完整路径
            new_jpg_path = os.path.join(jpg_dir, new_jpg_filename)

            # 只重命名JPG文件
            try:
                os.rename(original_jpg_path, new_jpg_path)
                print(f"无匹配TXT: [{jpg_filename}] -> 重命名为 [{new_jpg_filename}]")
                no_match_count += 1
            except OSError as e:
                print(f"重命名文件时发生错误: {e}")

        processed_count += 1

    print("\n--------------------")
    print("处理完成！")
    print(f"总共处理了 {processed_count} 个JPG文件。")
    print(f"成功匹配并重命名 {match_count} 对文件。")
    print(f"无匹配并重命名 {no_match_count} 个JPG文件。")

if __name__ == '__main__':
    # 运行主函数
    rename_files(jpg_folder_path, txt_folder_path)
