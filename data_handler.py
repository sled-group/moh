import os
import json

class DataHandler:
    def __init__(self, base_dir):
        """
        Initialize the DataHandler.

        Args:
            base_dir (str): Base directory containing 'train' and 'validation' folders.
        """
        self.base_dir = base_dir

    def load_data(self, data_type):
        """
        Load data from the specified directory.

        Args:
            data_type (str): Type of data to load ('train' or 'validation').

        Returns:
            dict: Loaded data.
        """
        data_dir = os.path.join(self.base_dir, data_type)
        json_files = [
            "AAAAB_data.json",
            "BAAAA_data.json",
            "merged_heterogenous_data.json",
            "merged_homogenous_data.json",
            "merged_mixed_data.json"
        ]

        data = {}
        for json_file in json_files:
            file_path = os.path.join(data_dir, json_file)
            with open(file_path, "r") as f:
                data[json_file] = json.load(f)
        return data
    
    def file2folder(self, file_name):
        """
        Map file names to folder names.

        Args:
            file_name (str): Name of the file.

        Returns:
            str: Corresponding folder name.
        """
        file2folder_dict = {
            "AAAAB_data.json": "AAAAB",
            "BAAAA_data.json": "BAAAA",
            "merged_heterogenous_data.json": "heterogenous",
            "merged_homogenous_data.json": "homogenous",
            "merged_mixed_data.json": "mixed"
        }
        return file2folder_dict[file_name]
    
    def create_directories_for_file(self, file_path):
        """
        Create all intermediate directories for the given file path.

        Args:
            file_path (str): The file path for which to create directories.
        """
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

    def save_partial_results(self, processed_data, metrics, photo2answer, output_path, file_name):
        """
        Save partial results to JSON files after processing each entry.

        Args:
            processed_data (list): Processed data.
            metrics (dict): Evaluation metrics including accuracy, F1 score, precision, and recall.
            photo2answer (dict): Mapping of photo paths to predictions.
            output_path (str): Output directory.
            file_name (str): Name of the JSON file being processed.
        """
        folder = self.file2folder(file_name)
        output_path = os.path.join(output_path, folder)
        os.makedirs(output_path, exist_ok=True)
        
        output_file_path = os.path.join(output_path, file_name)
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(processed_data, file, indent=4)
        print(f"Results saved to {output_file_path}")

        metrics_path = os.path.join(output_path, f"{folder}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)
        print(f"Metrics saved to {metrics_path}")

        photo2answer_path = os.path.join(output_path, f"{folder}_photo2answer.json")
        with open(photo2answer_path, "w", encoding="utf-8") as file:
            json.dump(photo2answer, file, indent=4)
        print(f"Photo2Answer mapping saved to {photo2answer_path}")

    def save_final_results(self, all_results, all_metrics, all_photo2answer, output_path):
        """
        Save final results to JSON files after processing all entries.

        Args:
            all_results (dict): All processed data results.
            all_metrics (dict): All evaluation metrics.
            all_photo2answer (dict): All photo2answer mappings.
            output_path (str): Output directory.
        """
        for file_name, result_data in all_results.items():
            folder = self.file2folder(file_name)
            output_dir = os.path.join(output_path, folder)
            os.makedirs(output_dir, exist_ok=True)
            
            output_file_path = os.path.join(output_dir, file_name)
            self.create_directories_for_file(output_file_path)
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(result_data, file, indent=4)
            print(f"Final results saved to {output_file_path}")

            metrics_path = os.path.join(output_dir, f"{folder}_final_metrics.json")
            self.create_directories_for_file(metrics_path)
            with open(metrics_path, "w", encoding="utf-8") as file:
                json.dump(all_metrics[file_name], file, indent=4)
            print(f"Metrics saved to {metrics_path}")

            photo2answer_path = os.path.join(output_dir, f"{folder}_final_photo2answer.json")
            self.create_directories_for_file(photo2answer_path)
            with open(photo2answer_path, "w", encoding="utf-8") as file:
                json.dump(all_photo2answer[file_name], file, indent=4)
            print(f"Photo2Answer mapping saved to {photo2answer_path}")