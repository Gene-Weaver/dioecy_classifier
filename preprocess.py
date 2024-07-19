from datasets import DatasetDict, Dataset, Image, ClassLabel, Features, load_from_disk
import os
from PIL import Image as PILImage, ImageFile
from tqdm import tqdm


'''
This is used to parition the training data and then upload the dataset to hugging face



huggingface-cli login

dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...

'''

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_and_upload_hf_dataset(dataset_dir, dataset_name, dataset_path, private=True):
    os.makedirs(dataset_path, exist_ok=True)

    # Define the splits typically included in the dataset
    splits = ['train', 'test']  # Include 'validation' if it's used
    dataset_dict = {}

    # Process each split
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue

        class_names = sorted(os.listdir(split_dir))
        images = []
        labels = []

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            class_index = class_names.index(class_name)
            for image_name in tqdm(os.listdir(class_dir), desc=f"Processing {split}/{class_name}"):
                image_path = os.path.join(class_dir, image_name)
                images.append(image_path)
                labels.append(class_index)

        # Define the features of the dataset
        features = Features({
            'image': Image(),
            'label': ClassLabel(num_classes=len(class_names), names=class_names)
        })
        
        # Create Dataset for this split
        dataset = Dataset.from_dict({
            'image': images,
            'label': labels
        }, features=features)
        dataset_dict[split] = dataset

    # Create a dataset dictionary from the assembled datasets
    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict.save_to_disk(dataset_path)
    
    # Push to Hub
    dataset_dict.push_to_hub(dataset_name, private=private)

def print_image_labels(dataset):
    for image_dict in tqdm(dataset, desc="Printing image names and labels"):
        file_path = image_dict['image']
        class_label = image_dict['label']
        class_names = dataset.features['label'].names
        class_name = class_names[class_label]
        # Extracting the filename from the file path
        print(f"Image: Class Label: {class_label}, Class Name: {class_name}")


def main():
    '''
    ### FIRST
    huggingface-cli login
    '''
    
    # dataset_dir = '/home/brlab/Dropbox/SwinV2_Classifier/data/training_v_1'
    # dataset_name = 'dioecy_Dioscorea_v-1-1'
    # dataset_path = '/home/brlab/Dropbox/SwinV2_Classifier/data/training_v_1/dataset_dict'

    dataset_dir = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3'
    dataset_name = 'dioecy_Dioscorea_v-1-3'
    dataset_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3/dataset_dict'

    create_and_upload_hf_dataset(dataset_dir, dataset_name, dataset_path)
    

    # After dataset creation, to print out the class to id mapping, load the dataset from disk
    dataset_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3/dataset_dict'
    # dataset_path = '/home/brlab/Dropbox/SwinV2_Classifier/data/training_v_3/dataset_dict'

    # Load the dataset from the saved path
    dataset_dict = load_from_disk(dataset_path)

    print("\nTest Dataset:")
    print_image_labels(dataset_dict['test'])

    # Get the 'train' split dataset
    train_dataset = dataset_dict['train']

    # Get the class names from the 'label' feature
    class_names = train_dataset.features['label'].names

    # Now print out the mapping
    print("Class name to ID mapping:")
    for id, class_name in enumerate(class_names):
        print(f"{class_name}: {id}")

if __name__ == "__main__":
    main()