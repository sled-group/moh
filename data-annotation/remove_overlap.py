import json
import os
import glob
from joblib import Parallel, delayed

def calculate_iou(boxA, boxB):
    boxA = {k: int(v) for k, v in boxA.items()}
    boxB = {k: int(v) for k, v in boxB.items()}

    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA['xmax'] - boxA['xmin']) * (boxA['ymax'] - boxA['ymin'])
    boxBArea = (boxB['xmax'] - boxB['xmin']) * (boxB['ymax'] - boxB['ymin'])
    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / float(unionArea)

    return iou

def remove_excessive_overlaps(objects, threshold):
    objects.sort(key=lambda obj: (int(obj['bndbox']['xmax']) - int(obj['bndbox']['xmin'])) *
                                 (int(obj['bndbox']['ymax']) - int(obj['bndbox']['ymin'])))
    filtered_objects = []
    while objects:
        current_obj = objects.pop(0)
        keep = True
        for other_obj in filtered_objects:
            if calculate_iou(current_obj['bndbox'], other_obj['bndbox']) >= threshold:
                keep = False
                break
        if keep:
            filtered_objects.append(current_obj)
    return filtered_objects

def process_json_file(json_file_path, output_folder,threshold):
    output_file_path = os.path.join(output_folder, os.path.basename(json_file_path))
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        data['objects'] = remove_excessive_overlaps(data['objects'],threshold)
        if data['objects']:
            with open(output_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        else:
            print(f"No bounding boxes left in file: {os.path.basename(json_file_path)}, it will not be saved.")
    except Exception as e:
        print(f"Error processing file {json_file_path}: {e}")

def process_directory_parallel(input_dir, output_dir,threshold):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    Parallel(n_jobs=-1)(delayed(process_json_file)(json_file, output_dir,threshold) for json_file in json_files)

input_dir = 'data_json'  
output_dir = 'data_without_0.1_overlap'
threshold=0.1  #maximum overlap ratio
process_directory_parallel(input_dir, output_dir,threshold)
