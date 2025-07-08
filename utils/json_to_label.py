import os
import json

def load_classes(classes_path):
    with open(classes_path,'r',encoding='utf-8') as f:
        return [x.strip() for x in f if x.strip()]

def convert_points(points, width, height):
    norm = []
    for x, y in points:
        norm.append(round(x / width, 6))
        norm.append(round(y / height, 6))
    return norm

def labelme2yolo_segment(json_path, classes, output_txt):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_w, img_h = data['imageWidth'], data['imageHeight']

    lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in classes:
            continue
        class_id = classes.index(label)
        if shape.get('shape_type', 'polygon') != 'polygon':
            continue
        points = shape['points']
        norm_pts = convert_points(points, img_w, img_h)
        if len(norm_pts) < 6:
            continue
        line = str(class_id) + " " + " ".join(map(str, norm_pts))
        lines.append(line)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line+'\n')

def batch_labelme_folder_to_yolo_segment(input_dir, classes_txt, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    classes = load_classes(classes_txt)
    for fname in os.listdir(input_dir):
        if fname.endswith('.json'):
            json_path = os.path.join(input_dir, fname)
            out_name = os.path.splitext(fname)[0]+'.txt'
            out_path = os.path.join(output_dir, out_name)
            try:
                labelme2yolo_segment(json_path, classes, out_path)
                print(f"Converted: {fname} --> {out_name}")
            except Exception as e:
                print(f"Failed: {fname}: {e}")


if __name__ == '__main__':
    input_dir = '/home/jasperxzy/racing/racing_segmentation/dataset/raw_labels'         
    classes_txt = '/home/jasperxzy/racing/racing_segmentation/dataset/classes.txt'      
    output_dir = '/home/jasperxzy/racing/racing_segmentation/dataset/labels'             

    batch_labelme_folder_to_yolo_segment(input_dir, classes_txt, output_dir)
