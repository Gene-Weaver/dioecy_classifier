import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import pandas as pd
import argparse
import yaml

from LeafMachine2.leafmachine2.machine.machine_censor_components import machine

'''
IMG_RES = 512 for ResNet_3_1
IMG_RES = 1024 for ResNet_3_2
'''
IMG_RES = 512
N_CLASSES = 6



class ImagePredictor:
    def __init__(self, model_path, num_classes=N_CLASSES):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((IMG_RES, IMG_RES)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        # Update model_path_full to include the base directory
        base_dir = os.path.dirname(__file__)
        model_path_full = os.path.join(base_dir, self.model_path)

        self.model.load_state_dict(torch.load(model_path_full))
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_path):
        """Predict the class of a single image."""
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        return predicted.item()


def process_directory(input_dir, predictor):
    results = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                prediction = predictor.predict_image(image_path)
                results.append({
                    'image_name': file,
                    'full_path': image_path,
                    'prediction': prediction
                })

    output_csv = os.path.basename(input_dir.rstrip('/\\')) + '_predictions.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def run_censoring(config_path):
    # Run the LeafMachine2 censoring process
    dir_home = os.path.dirname(config_path)
    machine(config_path, dir_home, None)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'LeafMachine2', 'CensorArchivalComponents.yaml')

    # Load the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Get the input directory and model path from the configuration
    input_dir = config['leafmachine']['project']['dir_images_local']
    model_path = config.get('model_path', '/models/ResNet_v_3_1.pth')

    # Run the censoring process
    run_censoring(config_path)

    # The censored output directory becomes the input for ImagePredictor
    censored_output_dir = os.path.join(
        config['leafmachine']['project']['dir_output'], 
        config['leafmachine']['project']['run_name']
    )

    # Predict using the censored images
    predictor = ImagePredictor(model_path=model_path)
    process_directory(censored_output_dir, predictor)