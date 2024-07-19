import os
from transformers import AutoImageProcessor, Swinv2ForImageClassification, TrainingArguments, Trainer
import torch
from datasets import load_dataset, DatasetDict
import wandb
from PIL import Image as PILImage, ImageFile
from transformers import AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import Swinv2ForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt  # Fix import
import seaborn as sns
from collections import Counter
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation
from datasets import load_dataset, concatenate_datasets

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True



class CustomSwinv2ForImageClassification(Swinv2ForImageClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.3)
        self.class_weights = class_weights

    def forward(self, pixel_values=None, labels=None):
        outputs = super().forward(pixel_values=pixel_values)
        logits = self.dropout(outputs.logits)

        loss = None
        if labels is not None:
            # If class weights are provided, use them
            if self.class_weights is not None:
                class_weights = self.class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (logits, outputs.hidden_states, outputs.attentions, loss,)


def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def check_class_distribution(dataset):
    class_counts = {}
    for example in dataset:
        label = example['label']
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    print("Class distribution in the dataset:", class_counts)
    return class_counts

def calculate_class_weights(dataset):
    labels = [example['label'] for example in dataset]
    class_counts = Counter(labels)
    total_count = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {class_id: total_count / (num_classes * count) for class_id, count in class_counts.items()}
    
    weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float)
    return weights

def preprocess_data(examples):
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize((256, 256)),
            transforms.RandomApply([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image)

    processed_images = [preprocess_image(img) for img in examples['image']]
    return {'pixel_values': torch.stack(processed_images), 'labels': examples['label']}

def oversample_dataset(dataset):
    class_counts = Counter([example['label'] for example in dataset])
    max_count = max(class_counts.values())

    oversampled_datasets = []
    for class_id, count in class_counts.items():
        class_dataset = dataset.filter(lambda example: example['label'] == class_id)
        oversample_count = max_count - count
        if oversample_count > 0:
            oversample_indices = list(range(count)) * (oversample_count // count) + list(range(oversample_count % count))
            oversample_dataset = class_dataset.select(oversample_indices)
            oversampled_datasets.append(concatenate_datasets([class_dataset, oversample_dataset]))
        else:
            oversampled_datasets.append(class_dataset)
    
    return concatenate_datasets(oversampled_datasets).shuffle(seed=42)

def train_swin(local_dataset_dir):
    

    # Optional: Initialize W&B manually if you want more control over the configuration
    wandb.init(
        project="SwinV2",
        entity=os.getenv('WANDB_KEY'),  # Set your W&B entity here if required
        config={
            "learning_rate": 2e-4,
            "epochs": 40,
            "batch_size": 32
        }
    )
    # # Load your custom dataset
    # if os.path.exists(local_dataset_dir):
    # # Load dataset from local files if exists
    #     dataset = DatasetDict.load_from_disk(local_dataset_dir)
    # else:
    #     # Load dataset from Hugging Face Hub if local is not available
    dataset = load_dataset("phyloforfun/dioecy_Dioscorea_v-1-1")
    print("Train Dataset Sample:", dataset['train'][0])  # Check a sample for structure

    # train_class_counts = check_class_distribution(dataset['train'])
    # test_class_counts = check_class_distribution(dataset['test'])

    # Calculate class weights
    # class_weights = calculate_class_weights(dataset['train'])

    # Process the images and prepare them
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
    # model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
    model = CustomSwinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"
        # class_weights=class_weights
    )


    # Adjust the model to output four classes instead of 1000
    model.config.num_labels = 7  # Ensure the configuration knows how many labels there are
    model.classifier = torch.nn.Linear(model.classifier.in_features, 7)


    # Oversample the training dataset
    oversampled_train_dataset = oversample_dataset(dataset['train'])
    oversampled_eval_dataset = oversample_dataset(dataset['test'])
    
    train_dataset = oversampled_train_dataset.map(preprocess_data, batched=True, remove_columns=oversampled_train_dataset.column_names)
    eval_dataset = oversampled_eval_dataset.map(preprocess_data, batched=True, remove_columns=oversampled_eval_dataset.column_names)
    # eval_dataset = dataset['test'].map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    sample_batch = next(iter(train_loader))
    print("Batch structure:", type(sample_batch), sample_batch.keys())
    print("Labels shape:", sample_batch['labels'].shape)

    # Define training arguments with W&B integration
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        num_train_epochs=wandb.config.epochs,
        weight_decay=0.01,
        report_to="wandb",
        run_name="swin_transformer_v2_custom_classification",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None  # Pass the custom compute_metrics function
    )

    # Train the model
    trainer.train()

    # Optionally, save the model for later use
    model.save_pretrained("./custom_swin_transformer_v2")

    # Push the trained model to the Hugging Face Hub as a private model
    model.push_to_hub("swin_transformer_v2__dioecy_Dioscorea_v-1", private=True)

    # After training, you can use the model to predict new images as in your original code setup

if __name__ == '__main__':
    local_dataset_dir = '/home/brlab/Dropbox/SwinV2_Classifier/data/training_v_1/dataset_dict'
    train_swin(local_dataset_dir)


'''
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
--cfg /home/brlab/Dropbox/Swin-Transformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml --data-path /home/brlab/Dropbox/SwinV2_Classifier/data/training_v_1 --batch-size 64

'''