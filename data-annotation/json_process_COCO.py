import json
import os
from joblib import Parallel, delayed

def process_image_annotations(image_info, annotations, categories, output_directory):
    image_dict = {
        "folder": "",  
        "filename": image_info['file_name'],
        "source": {
            "database": "COCO",
            "image_id": str(image_info['id'])
        },
        "size": {
            "width": str(image_info['width']),
            "height": str(image_info['height']),
            "depth": "3"  
        },
        "segmented": "0",
        "objects": []
    }

    for ann in annotations:
        category_name = categories[ann['category_id']]['name']
        obj = {
            "name": category_name,
            "object_id": str(ann['id']),
            "difficult": "0",
            "bndbox": {
                "xmin": str(int(ann['bbox'][0])),
                "ymin": str(int(ann['bbox'][1])),
                "xmax": str(int(ann['bbox'][0] + ann['bbox'][2])),
                "ymax": str(int(ann['bbox'][1] + ann['bbox'][3]))
            }
        }
        image_dict['objects'].append(obj)

    json_filename = os.path.splitext(image_info['file_name'])[0] + '.json'
    json_filepath = os.path.join(output_directory, json_filename)
    with open(json_filepath, 'w') as json_file:
        json.dump(image_dict, json_file, indent=2)

def create_json_for_each_image(coco_file_path, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    with open(coco_file_path, 'r') as file:
        coco_data = json.load(file)

    categories = {category['id']: category for category in coco_data['categories']}

    Parallel(n_jobs=-1)(delayed(process_image_annotations)(
        image_info,
        [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']],
        categories,
        output_directory
    ) for image_info in coco_data['images'])

# Define root and output directories
coco_file_path = 'annotations/instances_train2017.json'
output_directory = 'data_json/'
create_json_for_each_image(coco_file_path, output_directory)
