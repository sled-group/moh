import json
import os
import glob
import random

def filter_top_5_bbox_and_save(json_file, output_folder):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        for obj in data['objects']:
            bndbox = obj['bndbox']
            obj['area'] = (int(bndbox['xmax']) - int(bndbox['xmin'])) * (int(bndbox['ymax']) - int(bndbox['ymin']))

        top_5_objects = sorted(data['objects'], key=lambda x: x['area'], reverse=True)[:5]
        for index, obj in enumerate(top_5_objects, start=1):
            obj['bbox_number'] = index  
            del obj['area'] 

        data['objects'] = top_5_objects

        original_filename = os.path.basename(json_file)
        json_filename = os.path.splitext(original_filename)[0] + '.json'
        output_file = os.path.join(output_folder, json_filename)

        with open(output_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except Exception as e:
        print(f"Error processing file {json_file}: {e}")

def shuffle_bbox_numbers_in_folder(folder_path):
    json_files = glob.glob(f"{folder_path}/*.json")

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
        if 'objects' in data:
            random.shuffle(data['objects'])
            for i, obj in enumerate(data['objects']):
                obj['bbox_number'] = i + 1
        with open(json_file, 'w') as file:
            json.dump(data, file, indent=4)
            
# Folder paths
source_folder = '1000_0.01'  # Path to the original JSON folder
output_folder = '1000_0.01_5'  # Path to the output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

json_files = glob.glob(os.path.join(source_folder, '*.json'))

# Process and save each JSON file
for json_file in json_files:
    filter_top_5_bbox_and_save(json_file, output_folder)

shuffle_bbox_numbers_in_folder(output_folder)
