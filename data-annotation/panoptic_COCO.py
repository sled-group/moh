import json
import os
import numpy as np
from PIL import Image

def load_labels(label_file_path):
    labels = {}
    with open(label_file_path, 'r') as file:
        for line in file:
            index, label = line.strip().split(': ')
            labels[int(index)] = label
    return labels

def match_indices_to_labels(unique_indices, labels):
    matched_labels = {index: labels.get(index, 'Unknown') for index in unique_indices}
    return matched_labels

def add_object_set_to_images(merged_json_path, labels_txt_path, images_directory, output_json_path):
    # Load the class labels from 'labels.txt'
    labels = load_labels(labels_txt_path)

    # Load the merged images data from the json file
    with open(merged_json_path, 'r') as file:
        images_data = json.load(file)

    for image in images_data:
        # Extract the filename without the extension and construct the path to the .png file
        if image['data_source']!='COCO':
            continue
        filename_without_ext = os.path.splitext(image['filename'])[0]
        image_file_path = os.path.join(images_directory, filename_without_ext + '.png')

        # If the .png file exists, read it and extract label indices
        if os.path.exists(image_file_path):
            with Image.open(image_file_path) as img:
                img_data = np.array(img)
            unique_classes = np.unique(img_data)
            adjusted_classes = unique_classes + 1
            adjusted_classes = adjusted_classes[(adjusted_classes != 255) & (adjusted_classes != 0)]
            # Match the indices to their labels
            label_names = [labels.get(index) for index in adjusted_classes if index in labels]
            # Update the 'object_set' in the image data
            image['object_set'].extend(label_names)
            # Remove duplicates
            image['object_set'] = list(set(image['object_set']))

    # Write the updated images data to the new json file
    with open(output_json_path, 'w') as file:
        json.dump(images_data, file, indent=4)

# Usage example
merged_json_path = 'merged_data.json'  # Path to the original merged_data.json
labels_txt_path = '/cocostuff/labels.txt'  # Path to the labels.txt file
images_directory = '/annotations/val2017'  # Directory where the .png files are located
output_json_path = 'merged_data_updated.json'  # Path for the new updated JSON file

# Run the function to update the merged_data.json with new object classes
add_object_set_to_images(merged_json_path, labels_txt_path, images_directory, output_json_path)
