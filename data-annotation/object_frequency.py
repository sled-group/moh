import json
from collections import Counter

def count_object_classes(merged_json_path, output_txt_path):
    with open(merged_json_path, 'r') as file:
        images_data = json.load(file)

    name_counter = Counter()
    for image in images_data:
        for obj in image['objects']:
            name_counter[obj['name']] += 1

    with open(output_txt_path, 'w') as file:
        total_classes = len(name_counter)
        file.write(f"Total different classes: {total_classes}\n")
        for name, count in name_counter.items():
            file.write(f"{name}: {count}\n")

merged_json_path = 'merged_data_json.json' 
output_txt_path = 'object_frequency.txt'
count_object_classes(merged_json_path, output_txt_path)