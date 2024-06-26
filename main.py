import os
from tqdm import tqdm
import argparse
from data_handler import DataHandler
from evaluation import get_evaluation
from image_processor import ImageProcessor
from model_handler import get_model

def main(model_name, model_size, model_path, device_map, data_base_path, output_base_path):
    # Initialize model handler, data handler, and image processor
    model = get_model(model_name, model_size, model_path, device_map)
    data_handler = DataHandler(data_base_path)
    image_processor = ImageProcessor()

    # Evaluation settings and data types
    settings = ["default", "student-forcing", "teacher-forcing", "single"]
    data_types = ["train", "validation"]

    # Iterate through each setting and data type
    for setting in tqdm(settings, desc="Settings"):
        for data_type in tqdm(data_types, desc=f"Data Types for {setting}", leave=False):
            # Load the data for the current data type
            data = data_handler.load_data(data_type)
            
            # Get the evaluation class for the current setting
            evaluation_class = get_evaluation(setting)
            
            # Define the output path
            output_path = os.path.join(output_base_path, model_name + model_size, setting, data_type)
            
            # Process each data file
            for file_name, entries in tqdm(data.items(), desc=f"Files for {data_type}", leave=False):
                processed_data = []
                photo2answer = {}
                acc_list = [0, 0, 0, 0, 0]
                evaluator = evaluation_class(model, image_processor, data_base_path, data_type)

                try:
                    for entry in tqdm(entries, desc=f"Entries for {file_name}", leave=False):
                        evaluator.process_entry(entry, acc_list, photo2answer)
                        processed_data.append(entry)

                        # Save partial results after processing each entry
                        metrics = evaluator.calculate_metrics(acc_list)
                        data_handler.save_partial_results(processed_data, metrics, photo2answer, output_path, file_name)
                    # data_handler.save_final_results(processed_data, metrics, photo2answer, output_path, file_name)
                except Exception as e:
                    print(f"Error processing file {file_name} in setting {setting}, data type {data_type}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model_size', type=str, required=True, help='Size of the model to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--device_map', type=str, required=True, help='Device map to use for model')
    parser.add_argument('--data_base_path', type=str, required=True, help='Base path for data')
    parser.add_argument('--output_base_path', type=str, required=True, help='Base path for output')

    args = parser.parse_args()

    main(args.model_name, args.model_size, args.model_path, args.device_map, args.data_base_path, args.output_base_path)