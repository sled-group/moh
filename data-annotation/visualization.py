import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
from joblib import Parallel, delayed
import glob

# Load local image from path
def load_image_from_path(image_path):
    try:
        img = PIL_Image.open(image_path)
        return img
    except IOError as e:
        print(f"Error opening image from {image_path}: {e}")
        return None

# Visualize bounding boxes and object numbers
def visualize_objects(image_info, image_path, output_path):
    img = load_image_from_path(image_path)
    if img is None:
        return

    plt.figure(frameon=False)
    plt.imshow(img)
    ax = plt.gca()

    for obj in image_info['objects']:
        bndbox = obj['bndbox']
        xmin = int(bndbox['xmin'])
        ymin = int(bndbox['ymin'])
        width = int(bndbox['xmax']) - xmin
        height = int(bndbox['ymax']) - ymin
        ax.add_patch(Rectangle((xmin, ymin), width, height, fill=False, edgecolor='red', linewidth=2))
        bbox_number = obj.get('bbox_number', 'N/A')
        ax.text(xmin, ymin-10, f'Obj.{bbox_number}', style='italic', color='white', fontsize=12,
                bbox={'facecolor': 'black', 'alpha': 0.75, 'pad': 3})

    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process and visualize images based on JSON file
def process_json_file(json_file, output_folder):
    try:
        with open(json_file, 'r') as file:
            image_info = json.load(file)
        image_folder = image_info['folder']
        image_filename = image_info['filename']
        image_path = os.path.join(image_folder, image_filename)
        output_path = os.path.join(output_folder, os.path.splitext(image_filename)[0] + '.png')
        visualize_objects(image_info, image_path, output_path)
    except Exception as e:
        print(f"Error processing file {json_file}: {e}")

def process_images_parallel(json_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    Parallel(n_jobs=-1)(delayed(process_json_file)(json_file, output_folder) for json_file in json_files)

# Example usage
json_folder = 'path/to/json/folder'  # Directory containing JSON files
output_folder = 'path/to/output/visualized/images'  # Output directory for visualized images

process_images_parallel(json_folder, output_folder)
