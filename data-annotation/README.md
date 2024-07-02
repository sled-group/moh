# Data Curation for ROPE

## Overview

This project contains various Python scripts designed for processing and analyzing datasets, particularly focusing on the COCO and ADE datasets. Below is a brief description of each Python script and its functionality.

## File Descriptions

1. **json_process_ADE.py**
   - **Description:** This script processes the raw data of ADE dataset into JSON files. 

2. **json_process_COCO.py**
   - **Description:** This script processes the raw data of COCO dataset into JSON files. 

3. **merge.py**
   - **Description:** Merges JSON files from different images into a single consolidated JSON file.

4. **object_frequency.py**
   - **Description:** Calculates the frequency of different object classes within the dataset.

5. **select_top50.py**
   - **Description:** Retains the top 50 most frequent object classes in the dataset.
  
6. **remove_overlap.py**
   - **Description:** Removes bounding boxes that overlap above a specified IoU threshold to ensure clean data.

7. **select_1000.py**
   - **Description:** This script selects the 1000 data points from the total data.

8. **object_set.py**
   - **Description:** Adds an "object_set" attribute to all images, facilitating the identification and listing of all distinct objects present in each image.

9. **panoptic_COCO.py**
   - **Description:** This script processes the panoptic segmentation data from the COCO dataset. It handles the specific annotations and formatting required for panoptic analysis, which completes the "object_set" in the COCO data.

10. **visualization.py**
    - **Description:** Provides visualization tools for the dataset, allowing for the visualization of selected bounding boxes in each image.
   

## Data Source

1. **COCO dataset**
   - COCO2017 training and validation set
   - COCO dataset: https://cocodataset.org/#explore
   - COCO panoptic dataset: https://github.com/nightrome/cocostuff

2. **ADE dataset**
   - ADE20K training and validation set
   - ADE20K dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/


## Script Instruction
- To process the datasets, first run **json_process.py** to generate JSON files. Then, run **merge.py** to create a merged JSON file for each dataset. Next, run **object_frequency.py** to calculate and sort the object frequency of each class. Subsequently, run **select_top50.py** to retain only the bounding boxes of the top 50 most frequent classes. Afterward, run **remove_overlap.py** to remove redundant bounding boxes. Then, run **select_1000.py** to select 1000 images from the total image pool. If a panoptic object set is needed, run **object_set.py** followed by the additional **panoptic_COCO.py** for the COCO dataset. Finally, run **visualization.py** to visualize the bounding boxes in each image.

Please submit issues and we will try our best to fix them!
