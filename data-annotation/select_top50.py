import os
import json
from joblib import Parallel, delayed
import glob

def get_top_categories(file_path, top_n):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        top_categories = [line.split(':')[0].strip() for line in lines[:top_n]]
    return set(top_categories)

def process_json_file(json_file_path, output_folder, top_categories):
    filename = os.path.basename(json_file_path)
    new_json_file_path = os.path.join(output_folder, filename)

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        filtered_objects = [obj for obj in data['objects'] if obj['name'] in top_categories]
        if filtered_objects:
            data['objects'] = filtered_objects
            with open(new_json_file_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)
    except Exception as e:
        print(f"Error processing file {json_file_path}: {e}")

def process_directory_parallel(input_dir, output_dir, top_categories):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    Parallel(n_jobs=-1)(delayed(process_json_file)(json_file, output_dir, top_categories) for json_file in json_files)

# Read the top 50 categories from object_frequency.txt
class_counts_file = 'object_frequency.txt'  # Path to the category count file
top_categories = get_top_categories(class_counts_file, 50)

input_dir = 'data_without_overlap'  # Directory where your JSON files are located
output_dir = 'data_without_overlap_top50_new'  # Directory where you want to save the new JSON files

process_directory_parallel(input_dir, output_dir, top_categories)