import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, wandb, random
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import OrderedDict

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagePredictor:
    def __init__(self, model_path, num_classes=7):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load pre-trained ResNeXt model
        self.model = models.resnext101_32x8d(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path))
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
    
def test_sweep(test_dir, predictor, save_name):
    class_names = sorted(os.listdir(test_dir))
    
    total_correct = 0
    total_images = 0
    
    # Results dictionary to store results for each class
    results = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    
    y_true = []
    y_pred = []
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        image_files = os.listdir(class_dir)
        
        # Sample roughly 10% of images from each class for the test sweep
        # num_samples = max(1, len(image_files) // 10)
        # sampled_images = random.sample(image_files, num_samples)
        sampled_images = image_files
        
        for image_name in tqdm(sampled_images, desc=f'Testing Class --- {class_name}'):
            image_path = os.path.join(class_dir, image_name)
            predicted_class_index = predictor.predict_image(image_path)
            predicted_class_name = class_names[predicted_class_index]
            
            y_true.append(class_name)
            y_pred.append(predicted_class_name)
            
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
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_name, dpi=300)
    # plt.show()



class ImageClassifier:
    def __init__(self, train_dir, val_dir, 
                num_classes=7, 
                batch_size=16, 
                num_epochs=10, 
                learning_rate=0.001, 
                model_path="./output/resnext101_32x8d.pth", 
                checkpoint_path=None, 
                save_every_n_epochs=5, 
                accuracy_thresholds=[0.65, 0.80],
                resume=False):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.save_every_n_epochs = save_every_n_epochs
        self.accuracy_thresholds = accuracy_thresholds
        self.resume = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the directory for model_path exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Initialize wandb
        wandb.init(project="image-classification-resnext101_32x8d")

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load datasets
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.transform)

        # Calculate class weights
        class_counts = np.bincount([label for _, label in self.train_dataset])
        class_weights = 1. / class_counts
        samples_weights = class_weights[self.train_dataset.targets]
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Load pre-trained ResNeXt model
        self.model = models.resnext101_32x8d(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self.model.to(self.device)

        # Define loss function with class weights
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Enable mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize threshold tracking
        self.lr_reduced_thresholds = [False] * len(self.accuracy_thresholds)

        # Variables to track for resuming training
        self.start_epoch = 0

        # Resume from checkpoint if specified
        if self.resume:
            self.load_checkpoint()

    def train(self):
        # Training loop
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f'Epoch - {epoch + 1}/{self.num_epochs}')
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc='Training'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            wandb.log({"Training Loss": avg_loss, "epoch": epoch + 1})

            accuracy = self.validate(epoch + 1)
            print(f'    Val accuracy for epoch {epoch + 1}: {accuracy}')

            # Check if accuracy thresholds are reached and reduce learning rate
            for i, threshold in enumerate(self.accuracy_thresholds):
                if accuracy >= threshold and not self.lr_reduced_thresholds[i]:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10
                    print(f"    Learning rate reduced to {self.optimizer.param_groups[0]['lr']} at threshold {threshold}")
                    self.lr_reduced_thresholds[i] = True

            # Save the trained model every n epochs
            if (epoch + 1) % self.save_every_n_epochs == 0:
                checkpoint_path = f"{os.path.splitext(self.model_path)[0]}_epoch_{epoch + 1}.pth"
                self.save_checkpoint(epoch + 1)
                print(f"    Model saved to {checkpoint_path}")

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_path)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = correct / total
        wandb.log({"Validation Loss": avg_val_loss, "Accuracy": accuracy, "epoch": epoch})
        return accuracy

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'lr_reduced_thresholds': self.lr_reduced_thresholds
        }
        checkpoint_path = f"{os.path.splitext(self.model_path)[0]}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.checkpoint_path)
            print("Checkpoint keys:", checkpoint.keys())

            # Load model weights directly if the checkpoint is an OrderedDict of weights
            if isinstance(checkpoint, OrderedDict):
                self.model.load_state_dict(checkpoint)
                print("Loaded model state_dict directly from checkpoint.")
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.start_epoch = checkpoint['epoch']
                self.lr_reduced_thresholds = checkpoint['lr_reduced_thresholds']
                print(f"Loaded checkpoint from epoch {self.start_epoch}")

        except KeyError as e:
            print(f"KeyError: {e} - Checkpoint file does not contain the key.")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e} - Checkpoint file not found.")
        except Exception as e:
            print(f"Exception: {e} - An unexpected error occurred.")

    def predict(self):
        # Initialize wandb (optional)
        # wandb.init(project="image-classification-inference")

        # Load validation dataset
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Run inference
        predictions = []
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader, desc='Inference'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        # Log predictions to wandb
        class_names = self.val_dataset.classes
        for i, prediction in enumerate(predictions):
            print({"Image Index": i, "Predicted Class": class_names[prediction]})

        print("Inference completed.")

if __name__ == "__main__":
    do_train = True
    do_eval = True

    if do_train:
        classifier = ImageClassifier(
            train_dir='D:/Dropbox/SwinV2_Classifier/data/training_v_2/train',
            val_dir='D:/Dropbox/SwinV2_Classifier/data/training_v_2/test',
            model_path='D:/Dropbox/SwinV2_Classifier/ResNeXt_v_2_2/ResNeXt_v_2_2.pth',
            num_epochs=250,
            resume=True,
            checkpoint_path='D:/Dropbox/SwinV2_Classifier/ResNeXt_v_2/ResNeXt_v_2_epoch_50.pth',
        )
        classifier.train()
    '''
    ResNeXt_v_2 = val 43.82% train 73.53% (e50 total)
    ResNeXt_v_2_2 = val 64.94% train 99.64% (e50 + e250 total)
    '''
    if do_eval:
        print('Eval')
        predictor = ImagePredictor(model_path='D:/Dropbox/SwinV2_Classifier/ResNeXt_v_2_2/ResNeXt_v_2_2.pth')
        test_sweep('D:/Dropbox/SwinV2_Classifier/data/training_v_2/test', predictor, save_name='./ResNeXt_v_2_2/confusion_matrix_test_eComplete.png')
        test_sweep('D:/Dropbox/SwinV2_Classifier/data/training_v_2/train', predictor, save_name='./ResNeXt_v_2_2/confusion_matrix_train_eComplete.png')
