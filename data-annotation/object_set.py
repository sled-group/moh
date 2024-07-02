import json
import os

def collect_object_names(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        object_names = {obj['name'] for obj in data.get('objects', [])}
    return object_names

def add_object_set_to_images(merged_json_path, source_json_directory):
    with open(merged_json_path, 'r') as file:
        images_data = json.load(file)

    for image in images_data:
        filename_without_ext = os.path.splitext(image['filename'])[0]
        source_json_file = os.path.join(source_json_directory, filename_without_ext + '.json')
        if os.path.exists(source_json_file):
            object_names = collect_object_names(source_json_file)
            image['object_set'] = list(object_names)

    with open(merged_json_path, 'w') as file:
        json.dump(images_data, file, indent=4)

# Example usage
merged_json_path = 'merged_data.json'  # Path to merged_data.json
source_json_directory = 'data_json_validation'  # Directory containing individual JSON files
add_object_set_to_images(merged_json_path, source_json_directory)
