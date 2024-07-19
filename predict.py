import os, random
from transformers import AutoImageProcessor, Swinv2ForImageClassification, TrainingArguments, Trainer
import torch
from datasets import load_dataset, DatasetDict
import wandb
from PIL import Image as PILImage, ImageFile
from transformers import AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import Swinv2ForImageClassification

from train import CustomSwinv2ForImageClassification, calculate_class_weights

# Allow loading of truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
This was  used to test Swin-V2 but is not currently in use

both_both: 0
both_flower: 1
female_both: 2
female_flower: 3
female_fruit: 4
male_flower: 5
sterile: 6
'''

class IndexToClassNameConverter:
    def __init__(self):
        # This mapping is based on the manual mapping you provided in the comment block above.
        self.index_to_class_name = {
            0: "both_both",
            1: "both_flower",
            2: "female_both",
            3: "female_flower",
            4: "female_fruit",
            5: "male_flower",
            6: "sterile"
        }

    def get_class_name(self, index):
        return self.index_to_class_name.get(index, "Unknown")

class ImagePredictor:
    def __init__(self, model_path):
        self.model = CustomSwinv2ForImageClassification.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
        self.converter = IndexToClassNameConverter()

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path)
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        # Make a prediction
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs[0]  # Access the logits from the tuple

        # Log logits and softmax values for debugging
        softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
        print(f"Logits: {logits}")
        print(f"Softmax probabilities: {softmax_probs}")

        # Get the predicted class index
        predicted_class_index = torch.argmax(logits, dim=-1).item()
        # Convert index to class name
        predicted_class_name = self.converter.get_class_name(predicted_class_index)

        return predicted_class_name


def test_sweep(dataset_dir, predictor):
    # Assuming the dataset_dir is structured as specified with each class having its own folder
    test_dir = os.path.join(dataset_dir, 'val')
    class_names = sorted(os.listdir(test_dir))
    
    converter = IndexToClassNameConverter()
    total_correct = 0
    total_images = 0
    
    # Results dictionary to store results for each class
    results = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        image_files = os.listdir(class_dir)
        
        # Sample roughly 10% of images from each class for the test sweep
        num_samples = max(1, len(image_files) // 10)
        sampled_images = random.sample(image_files, num_samples)
        
        for image_name in sampled_images:
            image_path = os.path.join(class_dir, image_name)
            predicted_class_name = predictor.predict(image_path)
            
            # Check if the prediction matches the folder name (actual class)
            if class_name == predicted_class_name:
                results[class_name]['correct'] += 1
                total_correct += 1
            
            # Increase the total count for this class and overall
            results[class_name]['total'] += 1
            total_images += 1
    
    # Print out the score for each class and the overall accuracy
    print("Accuracy score for each class:")
    for class_name, result in results.items():
        accuracy = result['correct'] / result['total'] if result['total'] > 0 else 0
        print(f"{class_name}: {accuracy * 100:.2f}%")
    
    # Calculate and print overall accuracy
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")



def main():
    # # Initialize the predictor with the path to your trained model
    # predictor = ImagePredictor("./custom_swin_transformer_v2")

    # # Make a prediction on an image
    # predicted_class_name = predictor.predict("/home/brlab/Dropbox/SwinV2_Classifier/COL000287543.jpg")
    # print(f"The predicted class is: {predicted_class_name}")

    dataset_dir = '/home/brlab/Dropbox/SwinV2_Classifier/data/training_v_1'  # Replace with your actual dataset path
    model_path = "./custom_swin_transformer_v2"
    
    predictor = ImagePredictor(model_path)
    test_sweep(dataset_dir, predictor)

if __name__ == '__main__':
    main()