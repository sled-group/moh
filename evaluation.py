from PIL import Image
from utils import precision_score, recall_score, f1_score
from constant import TRAIN_CLASSES, VALIDATION_CLASSES

class EvaluationBase:
    def __init__(self, model_handler, image_processor, dataset_path, data_type):
        """
        Initialize the evaluation base class.

        Args:
            model_handler (BaseModelHandler): The model handler.
            image_processor (ImageProcessor): The image processor.
            dataset_path (str): Path to the dataset.
            data_type (str): Type of data being processed ('train' or 'validation').
        """
        self.model_handler = model_handler
        self.image_processor = image_processor
        self.dataset_path = dataset_path
        self.data_type = data_type
        self.correct_predictions = 0
        self.total_predictions = 0
        self.y_true = []
        self.y_pred = []

    def process_entry(self, entry, acc_list, photo2answer):
        """
        Process a single entry and classify objects in the image.

        Args:
            entry (dict): Data entry to process.
            acc_list (list): List to accumulate accuracy data.
            photo2answer (dict): Dictionary to map photo paths to predictions.

        Returns:
            None
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def calculate_metrics(self, acc_list):
        """
        Calculate evaluation metrics including accuracy, F1 score, precision, and recall.

        Returns:
            dict: Evaluation metrics.
        """
        accuracy = (self.correct_predictions / self.total_predictions) * 100 if self.total_predictions else 0
        metrics = {
            "estimated_accuracy": f"{accuracy:.2f}%"
        }

        for avg in ['macro', 'micro', 'weighted']:
            precision = precision_score(self.y_true, self.y_pred, average=avg) * 100
            recall = recall_score(self.y_true, self.y_pred, average=avg) * 100
            f1 = f1_score(self.y_true, self.y_pred, average=avg) * 100

            metrics.update({
                f"precision_{avg}": f"{precision:.2f}%",
                f"recall_{avg}": f"{recall:.2f}%",
                f"f1_score_{avg}": f"{f1:.2f}%",
            })

        metrics.update({
            "acc_list": acc_list,
            "num_processed_data": int(self.total_predictions / 5),
        })
        return metrics

    def _generate_prompt(self, obj_num):
        """
        Generate a prompt based on the dataset name, data type, and object number.

        Args:
            obj_num (int): Object number.

        Returns:
            str: Generated prompt.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def extract_classes(self, text):
        """
        Extract the classes following obj1, obj2, obj3, obj4, and obj5 from the given text.

        Args:
            text (str): The input text containing class information.

        Returns:
            list: A list of classes ordered by obj1, obj2, obj3, obj4, obj5.
        """
        classes = [''] * 5  # Initialize a list with 5 empty strings
        lines = text.split('\n')
        for line in lines:
            if 'obj1:' in line:
                classes[0] = line.split('obj1:')[1].strip().strip("'").replace(".", "").replace("'", "").replace("*", "").replace(" ", "").lower()
            elif 'obj2:' in line:
                classes[1] = line.split('obj2:')[1].strip().strip("'").replace(".", "").replace("'", "").replace("*", "").replace(" ", "").lower()
            elif 'obj3:' in line:
                classes[2] = line.split('obj3:')[1].strip().strip("'").replace(".", "").replace("'", "").replace("*", "").replace(" ", "").lower()
            elif 'obj4:' in line:
                classes[3] = line.split('obj4:')[1].strip().strip("'").replace(".", "").replace("'", "").replace("*", "").replace(" ", "").lower()
            elif 'obj5:' in line:
                classes[4] = line.split('obj5:')[1].strip().strip("'").replace(".", "").replace("'", "").replace("*", "").replace(" ", "").lower()
        return classes

class SingleObjectEvaluation(EvaluationBase):
    def _generate_prompt(self, obj_num, dataset_name):
        """
        Generate a prompt based on the dataset name, data type, and object number.

        Args:
            obj_num (int): Object number.

        Returns:
            str: Generated prompt.
        """
        if self.data_type == 'train':
            if dataset_name == "COCO":
                class_list = ", ".join(TRAIN_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(TRAIN_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        else:
            if dataset_name == "COCO":
                class_list = ", ".join(VALIDATION_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(VALIDATION_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

        return f"User: Select the single, most appropriate class for Obj.{obj_num} located within the red bounding box from the following list: {class_list}. Your response should consist solely of the class name that Obj.{obj_num} belongs to, formatted as only the class name, without any extra characters or punctuation."
    
    def process_entry(self, entry, acc_list, photo2answer):
        """
        Process a single entry and classify objects in the image.

        Args:
            entry (dict): Data entry to process.
            acc_list (list): List to accumulate accuracy data.
            photo2answer (dict): Dictionary to map photo paths to predictions.

        Returns:
            None
        """
        for index in range(1, 6):  # Each entry has 5 objects
            image_path = self.dataset_path + entry["folder"].replace("jpg", "png")
            image = Image.open(image_path)

            prompt = self._generate_prompt(index, entry["data_source"])
            processed_image = self.image_processor.preprocess_image(image)
            predicted_class = self.model_handler.generate_response(prompt, processed_image)
            entry["objects"][index - 1]["prediction"] = predicted_class
            photo2answer[image_path.split("/")[-1] + "-" + str(index)] = predicted_class

            true_class = entry["objects"][index - 1]["name"].lower()
            pred_class = predicted_class.lower()

            self.y_true.append(true_class)
            self.y_pred.append(pred_class)

            if pred_class == true_class:
                self.correct_predictions += 1
                acc_list[index - 1] += 1
        self.total_predictions += 5


class MultiObjectEvaluation(EvaluationBase):
    def segment_classes(self, response):
        """
        Segment out class labels from the given response string.

        Args:
            response (str): The response string in the format "obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>".

        Returns:
            list: A list of segmented class labels.
        """
        segments = response.split(", ")
        classes = [segment.split(": ")[1] for segment in segments]
        return classes
    
    def _generate_prompt(self, dataset_name):
        """
        Generate a prompt based on the dataset name, data type, and object number.

        Args:
            obj_num (int): Object number.

        Returns:
            str: Generated prompt.
        """
        prompt = "Given the classes: <50 classes>. There are five red bounding boxes in this image. For each object within the red bounding boxes, identify its class from the list. Provide the class names in the format: 'obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>', with no additional words or punctuation. For example: obj1: class, obj2: class, obj3: class, obj4: class, obj5: class. Replace class with the actual names of the classes from your class list. Ensure that no placeholders or brackets are used around the class names and that no additional words or punctuation are added to the response."
        
        if self.data_type == 'train':
            if dataset_name == "COCO":
                class_list = ", ".join(TRAIN_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(TRAIN_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        else:
            if dataset_name == "COCO":
                class_list = ", ".join(VALIDATION_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(VALIDATION_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

        return prompt.replace("<50 classes>", class_list)

    def process_entry(self, entry, acc_list, photo2answer):
        """
        Process a single entry and classify objects in the image.

        Args:
            entry (dict): Data entry to process.
            acc_list (list): List to accumulate accuracy data.
            photo2answer (dict): Dictionary to map photo paths to predictions.

        Returns:
            None
        """
        
        image_path = self.dataset_path + entry["folder"].replace("jpg", "png")
        image = Image.open(image_path)

        prompt = self._generate_prompt(entry["data_source"])
        processed_image = self.image_processor.preprocess_image(image)
        predicted_class = self.model_handler.generate_response(prompt, processed_image)
        photo2answer[image_path.split("/")[-1]] = predicted_class
        try:
            predicted_class = self.segment_classes(predicted_class)
            print("predicted_class: ", predicted_class)
        except Exception as e:
            self.total_predictions += 5
        
        for index in range(1, 6):  # Each entry has 5 objects
            if len(predicted_class) >= index:
                entry["objects"][index - 1]["prediction"] = predicted_class[index - 1]

                true_class = entry["objects"][index - 1]["name"].lower()
                pred_class = predicted_class[index - 1].lower()

                self.y_true.append(true_class)
                self.y_pred.append(pred_class)

                if pred_class == true_class:
                    self.correct_predictions += 1
                    acc_list[index - 1] += 1
        self.total_predictions += 5

class StudentForcingEvaluation(EvaluationBase):
    def segment_classes(self, response):
        """
        Segment out class labels from the given response string.

        Args:
            response (str): The response string in the format "obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>".

        Returns:
            list: A list of segmented class labels.
        """
        segments = response.split(", ")
        classes = [segment.split(": ")[1] for segment in segments]
        return classes
    
    def _generate_prompt(self, dataset_name):
        """
        Generate a prompt based on the dataset name, data type, and object number.

        Args:
            obj_num (int): Object number.

        Returns:
            str: Generated prompt.
        """
        prompt = "Given the classes: <50 classes>. There are five red bounding boxes in this image. For each object within the red bounding boxes, identify its class from the list. Provide the class names in the format: 'obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>', with no additional words or punctuation. For example: obj1: class, obj2: class, obj3: class, obj4: class, obj5: class. Replace class with the actual names of the classes from your class list. Ensure that no placeholders or brackets are used around the class names and that no additional words or punctuation are added to the response."
        
        if self.data_type == 'train':
            if dataset_name == "COCO":
                class_list = ", ".join(TRAIN_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(TRAIN_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        else:
            if dataset_name == "COCO":
                class_list = ", ".join(VALIDATION_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(VALIDATION_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

        return prompt.replace("<50 classes>", class_list)

    def process_entry(self, entry, acc_list, photo2answer):
        """
        Process a single entry and classify objects in the image.

        Args:
            entry (dict): Data entry to process.
            acc_list (list): List to accumulate accuracy data.
            photo2answer (dict): Dictionary to map photo paths to predictions.

        Returns:
            None
        """
        prompt = self._generate_prompt(entry["data_source"])
        for index in range(1, 6):  # Each entry has 5 objects
            image_path = self.dataset_path + entry["folder"].replace("jpg", "png")
            image = Image.open(image_path)

            if index == 1:
                cumulative_prompt = prompt + f" obj{index}:"
            else:
                cumulative_prompt = cumulative_prompt + f", obj{index}:"
            
            processed_image = self.image_processor.preprocess_image(image)
            predicted_class_str = self.model_handler.generate_response(cumulative_prompt, processed_image)
            try:
                predicted_class = self.segment_classes(predicted_class_str)
                print("predicted_class: ", predicted_class)
            except Exception as e:
                self.total_predictions += 5 - index + 1
                break
            
            photo2answer[image_path.split("/")[-1] + "-" + str(index)] = predicted_class_str
            if len(predicted_class) >= index:
                entry["objects"][index - 1]["prediction"] = predicted_class[index - 1]

                true_class = entry["objects"][index - 1]["name"].lower()
                pred_class = predicted_class[index - 1].lower()
                
                cumulative_prompt += " " + pred_class

                self.y_true.append(true_class)
                self.y_pred.append(pred_class)

                if pred_class == true_class:
                    self.correct_predictions += 1
                    acc_list[index - 1] += 1
            else:
                self.total_predictions += 5 - index + 1
                break
        self.total_predictions += 5

class TeacherForcingEvaluation(EvaluationBase):
    def segment_classes(self, response):
        """
        Segment out class labels from the given response string.

        Args:
            response (str): The response string in the format "obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>".

        Returns:
            list: A list of segmented class labels.
        """
        segments = response.split(", ")
        classes = [segment.split(": ")[1] for segment in segments]
        return classes
    
    def _generate_prompt(self, dataset_name):
        """
        Generate a prompt based on the dataset name, data type, and object number.

        Args:
            obj_num (int): Object number.

        Returns:
            str: Generated prompt.
        """
        prompt = "Given the classes: <50 classes>. There are five red bounding boxes in this image. For each object within the red bounding boxes, identify its class from the list. Provide the class names in the format: 'obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>', with no additional words or punctuation. For example: obj1: class, obj2: class, obj3: class, obj4: class, obj5: class. Replace class with the actual names of the classes from your class list. Ensure that no placeholders or brackets are used around the class names and that no additional words or punctuation are added to the response."
        
        if self.data_type == 'train':
            if dataset_name == "COCO":
                class_list = ", ".join(TRAIN_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(TRAIN_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        else:
            if dataset_name == "COCO":
                class_list = ", ".join(VALIDATION_CLASSES["COCO"])
            elif dataset_name == "ADE":
                class_list = ", ".join(VALIDATION_CLASSES["ADE"])
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

        return prompt.replace("<50 classes>", class_list)

    def process_entry(self, entry, acc_list, photo2answer):
        """
        Process a single entry and classify objects in the image.

        Args:
            entry (dict): Data entry to process.
            acc_list (list): List to accumulate accuracy data.
            photo2answer (dict): Dictionary to map photo paths to predictions.

        Returns:
            None
        """
        prompt = self._generate_prompt(entry["data_source"])
        for index in range(1, 6):  # Each entry has 5 objects
            image_path = self.dataset_path + entry["folder"].replace("jpg", "png")
            image = Image.open(image_path)

            if index == 1:
                cumulative_prompt = prompt + f" obj{index}:"
            else:
                cumulative_prompt = cumulative_prompt + f", obj{index}:"
            
            processed_image = self.image_processor.preprocess_image(image)
            predicted_class_str = self.model_handler.generate_response(cumulative_prompt, processed_image)
            try:
                predicted_class = self.segment_classes(predicted_class_str)
                print("predicted_class: ", predicted_class)
            except Exception as e:
                self.total_predictions += 5 - index + 1
                break
            
            photo2answer[image_path.split("/")[-1] + "-" + str(index)] = predicted_class_str
            if len(predicted_class) >= index:
                entry["objects"][index - 1]["prediction"] = predicted_class[index - 1]

                true_class = entry["objects"][index - 1]["name"].lower()
                pred_class = predicted_class[index - 1].lower()
                
                cumulative_prompt += " " + true_class

                self.y_true.append(true_class)
                self.y_pred.append(pred_class)

                if pred_class == true_class:
                    self.correct_predictions += 1
                    acc_list[index - 1] += 1
            else:
                self.total_predictions += 5 - index + 1
                break
        self.total_predictions += 5
    
def get_evaluation(setting):
    """
    Get the appropriate evaluation class based on the setting and data type.
    Args:
        setting (str): Evaluation setting.
        data_type (str): Type of data being processed ('train' or 'validation').

    Returns:
        class: Corresponding evaluation class.
    """
    if setting == "default":
        return MultiObjectEvaluation
    elif setting == "student-forcing":
        return StudentForcingEvaluation  # Example, change as needed
    elif setting == "teacher-forcing":
        return TeacherForcingEvaluation  # Example, change as needed
    elif setting == "single":
        return SingleObjectEvaluation
    else:
        raise ValueError(f"Unknown setting: {setting}")