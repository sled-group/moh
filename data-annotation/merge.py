import os
import json

def merge_json_files(source_directory, output_file):
    merged_data = []

    for file in os.listdir(source_directory):
        if file.endswith(".json"):
            file_path = os.path.join(source_directory, file)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                merged_data.append(data)

    with open(output_file, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)


source_directory = '/nfs/turbo/coe-chaijy-unreplicated/datasets/ADE20/temporary'  
output_file = '/nfs/turbo/coe-chaijy-unreplicated/datasets/ADE20/temporary_mixed.json' 
merge_json_files(source_directory, output_file)