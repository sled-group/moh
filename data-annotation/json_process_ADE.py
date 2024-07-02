import json
import os
import glob

def calculate_bounding_box(polygon):
    x_coordinates = polygon['x']
    y_coordinates = polygon['y']
    xmin = min(x_coordinates)
    xmax = max(x_coordinates)
    ymin = min(y_coordinates)
    ymax = max(y_coordinates)
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

def convert_json(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"Error reading file {input_file}: {e}")
        return

    try:
        converted_data = {
            "folder": data['annotation']['folder'],
            "filename": data['annotation']['filename'],
            "source": {"database": "Unknown", "image_id": "None", "coco_id": "None", "flickr_id": "None"},
            "size": {"width": str(data['annotation']['imsize'][1]), "height": str(data['annotation']['imsize'][0]), "depth": str(data['annotation']['imsize'][2])},
            "segmented": "0",
            "objects": [],
            "relations": []
        }

        for obj in data['annotation']['object']:
            bounding_box = calculate_bounding_box(obj['polygon'])
            object_data = {
                "name": obj['name'],
                "object_id": str(obj['id']),
                "difficult": "0",
                "bndbox": bounding_box
            }
            converted_data['objects'].append(object_data)

        with open(output_file, 'w') as file:
            json.dump(converted_data, file, indent=4)
    except (KeyError, IndexError) as e:
        print(f"Error processing file {input_file}: {e}")

def process_directory(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for json_file in glob.glob(os.path.join(root_dir, '**', '*.json'), recursive=True):
        output_file = os.path.join(output_dir, os.path.basename(json_file))
        convert_json(json_file, output_file)


# Define root and output directories
root_dir = 'ADE20K_2021_17_01/images/ADE/training'
output_dir = 'data_json' 

process_directory(root_dir, output_dir)
