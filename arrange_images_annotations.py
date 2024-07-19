import os, shutil
import pandas as pd
import numpy as np
from tqdm import tqdm  
from datasets import DatasetDict, Dataset, Image, ClassLabel, Features
from PIL import Image as PILImage, ImageFile
# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def rename_files(csv_path, folder_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if the column for filenames exists
    if 'file' not in df.columns:
        raise ValueError("CSV does not contain 'file' column")
    
    # Iterate over each file name in the DataFrame
    for index, row in df.iterrows():
        old_name = row['file']
        base_name, extension = os.path.splitext(old_name)
        
        # Replace periods with underscore and @ with double underscore in the base name
        new_base_name = base_name.replace('.', '_').replace('@', '__')
        new_name = new_base_name + extension  # Append the original extension to the new base name
        
        # Formulate the full old and new file paths
        old_file_path = os.path.join(folder_path, old_name)
        new_file_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
            df.at[index, 'file'] = new_name  # Update the file name in the dataframe
            print(f"Renamed {old_name} to {new_name}")
        else:
            print(f"File {old_name} does not exist in the specified folder")
    
    # Save the updated DataFrame back to CSV to reflect the new file names
    df.to_csv(csv_path, index=False)

def save_dataset_to_disk(dataset_path, data_folders, class_names):
    """
    Saves images and labels in the specified folders as a DatasetDict on disk.
    
    Args:
    dataset_path (str): Path where the DatasetDict will be saved.
    data_folders (dict): Dictionary with 'train' and 'test' keys pointing to their respective folders.
    class_names (list): List of class names corresponding to labels.
    """
    os.makedirs(dataset_path, exist_ok=True)  # Ensure the dataset directory exists
    
    dataset_dict = {}
    for split, folder in data_folders.items():
        images = []
        labels = []

        for class_index, class_name in enumerate(class_names):
            class_folder = os.path.join(folder, class_name)
            if not os.path.exists(class_folder):
                continue

            for image_filename in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_filename)
                try:
                    with PILImage.open(image_path) as img:
                        img.convert('RGB')  # Convert image to RGB
                        images.append(image_path)  # Storing path for simplicity
                        labels.append(class_index)
                except IOError:
                    print(f"Cannot open {image_path}. It may be corrupted.")

        # Create dataset for this split
        dataset = Dataset.from_dict({
            'image': images,
            'label': labels
        })
        dataset_dict[split] = dataset

    # Create and save DatasetDict
    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict.save_to_disk(dataset_path)
    print(f"Dataset saved to {dataset_path}")


# Custom converter function
def convert_to_bool(x):
    if pd.isna(x):
        return False
    try:
        # Convert to float first to handle both float and int '1' correctly
        return float(x) == 1
    except ValueError:
        # Handle strings
        return str(x).strip().lower() in ['1', 'y']

def classify_and_copy_images(csv_path_out, csv_path, images_folder_path, output_folder_path):
    # Check if the file exists and if it is empty
    file_exists = os.path.isfile(csv_path_out)
    file_empty = os.stat(csv_path_out).st_size == 0 if file_exists else False

    # Load the CSV file
    df = pd.read_csv(csv_path, converters={
        'fruits': convert_to_bool,
        'flowers': convert_to_bool,
        'reproduction.organ': convert_to_bool
    })
    df['sex'] = df['sex'].astype(str)
        
    
    # Add a 'type' and 'dataset' column if it doesn't exist
    if 'type' not in df.columns:
        df['type'] = None
    if 'dataset' not in df.columns:
        df['dataset'] = None

    # Create dictionaries to hold train and test file paths
    train_files = {}
    test_files = {}

    # Directory creation for training and testing
    train_path = os.path.join(output_folder_path, 'train')
    test_path = os.path.join(output_folder_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Iterate over each row in the DataFrame to classify and decide on train/test split
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        file_name = row['file']
        src_path = os.path.join(images_folder_path, file_name)

        # Classify the image
        if not row['reproduction.organ']: # or 'reproduction organ'
            class_name = 'sterile'
        else:
            part_1 = ""
            part_2 = ""
            if row['sex'].lower() == 'm':
                part_1 = 'male'
            elif row['sex'].lower() == 'f':
                part_1 = 'female'
            elif row['sex'].lower() == 'b':
                part_1 = 'both'
            else:
                continue  # If 'sex' is not one of 'm', 'f', 'b', skip this row

            # Determine part_2 based on 'flowers' and 'fruits'
            if part_1 == 'female':
                if row['phenophase'].lower() == 'l':
                    part_2 = 'flower'
                elif row['phenophase'].lower() == 'r':
                    part_2 = 'fruit'
            elif row['flowers'] and row['fruits']:
                part_2 = 'both'
            elif not row['flowers'] and not row['fruits']:
                part_2 = 'none'
            elif row['flowers']:
                part_2 = 'flower'
            elif row['fruits']:
                part_2 = 'fruit'
            else:
                continue  # If conditions are ambiguous, skip this row
            
            class_name = f"{part_1}_{part_2}"
        
        # Decide whether it should go into training or testing set (80/20 split)
        if np.random.rand() <= 0.8:
            dataset_type = 'train'
            dst_path = os.path.join(train_path, class_name, file_name)
        else:
            dataset_type = 'test'
            dst_path = os.path.join(test_path, class_name, file_name)

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy the file if it exists
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            df.at[index, 'type'] = class_name
            df.at[index, 'dataset'] = dataset_type
        elif os.path.exists(os.path.splitext(src_path)[0] + '.jpg'):
            src_path_jpg = os.path.splitext(src_path)[0] + '.jpg'
            dst_path_jpg = os.path.splitext(dst_path)[0] + '.jpg'
            shutil.copy(src_path_jpg, dst_path_jpg)
            df.at[index, 'file'] = os.path.basename(src_path_jpg)  # Update the DataFrame entry
            df.at[index, 'type'] = class_name
            df.at[index, 'dataset'] = dataset_type
        else:
            print(f"File {file_name} does not exist at {src_path}")

    # Save the updated DataFrame back to the specified output CSV
    df.to_csv(csv_path_out, mode='w', header=not file_exists or file_empty, index=False)



if __name__ == "__main__":
    '''Rename the files with @ and internal '.' '''
    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_Dpolygonoides.csv'
    # folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_polygonoides'
    # rename_files(csv_path, folder_path)

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_DioscoreaCol.csv'
    # folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_Colombia'
    # rename_files(csv_path, folder_path)

    '''Add files to training regimes v1 original images'''
    # csv_path_training_v_1 = 'D:/Dropbox/SwinV2_Classifier/data/training_v_1.csv'

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_DioscoreaCol.csv'
    # images_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_Colombia'
    # output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_1'
    # classify_and_copy_images(csv_path_training_v_1, csv_path, images_folder_path, output_folder_path)

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_DioscoreaHerbarium.csv'
    # images_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_HNCOL'
    # output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_1'
    # classify_and_copy_images(csv_path_training_v_1, csv_path, images_folder_path, output_folder_path)

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_Dpolygonoides.csv'
    # images_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_polygonoides'
    # output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_1'
    # classify_and_copy_images(csv_path_training_v_1, csv_path, images_folder_path, output_folder_path)

    # '''Save the dataset in the required format so you don't have to download from HF'''
    # data_folders = {
    #     'train': 'D:/Dropbox/SwinV2_Classifier/data/training_v_1/train',
    #     'test': 'D:/Dropbox/SwinV2_Classifier/data/training_v_1/test'
    # }
    # class_names = ['both_both','female_both','female_flower','female_fruit','male_flower','sterile'] 
    # dataset_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_1/dataset_dict'
    # save_dataset_to_disk(dataset_path, data_folders, class_names)




    '''Add files to training regimes v2 LM2 censored images'''
    # csv_path_training_v_2 = 'D:/Dropbox/SwinV2_Classifier/data/training_v_2.csv'

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_DioscoreaCol.csv'
    # images_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_Colombia_Censored'
    # output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_2'
    # classify_and_copy_images(csv_path_training_v_2, csv_path, images_folder_path, output_folder_path)

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_DioscoreaHerbarium.csv'
    # images_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_HNCOL_Censored'
    # output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_2'
    # classify_and_copy_images(csv_path_training_v_2, csv_path, images_folder_path, output_folder_path)

    # csv_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annot_Dpolygonoides.csv'
    # images_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/Dioscorea/annotated_images/Annotated_Dioscorea_polygonoides_Censored'
    # output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_2'
    # classify_and_copy_images(csv_path_training_v_2, csv_path, images_folder_path, output_folder_path)

    # '''Save the dataset in the required format so you don't have to download from HF'''
    # data_folders = {
    #     'train': 'D:/Dropbox/SwinV2_Classifier/data/training_v_2/train',
    #     'test': 'D:/Dropbox/SwinV2_Classifier/data/training_v_2/test'
    # }
    # class_names = ['both_both','female_both','female_flower','female_fruit','male_flower','sterile'] 
    # dataset_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_2/dataset_dict'
    # save_dataset_to_disk(dataset_path, data_folders, class_names)


    '''v3 '''
    # Step 1
    # csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_polygonoides/Phenoph_Annot_Dpolygonoides.csv'
    # folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_polygonoides/Annotated_Dioscorea_polygonoides'
    # rename_files(csv_path, folder_path)

    # csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_villosa-US/Phenoph_Annot_Dioscorea_villosa.csv'
    # folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_villosa-US/Annotated_Dioscorea_villosa'
    # rename_files(csv_path, folder_path)

    # csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-Colombia/Phenoph_Annot_DioscoreaColombia.csv'
    # folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-Colombia/Annotated_DioscoreaColombia'
    # rename_files(csv_path, folder_path)

    # csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-HNC-ColombianHerbarium/Phenoph_Annot_DioscoreaHerbarium.csv'
    # folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-HNC-ColombianHerbarium/Annotated_Dioscorea_HNCOL'
    # rename_files(csv_path, folder_path)

    # Step 2: censor the images, each folder to the Censored__
    # Step 3: below

    csv_path_training_v_3 = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3.csv'

    csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_polygonoides/Phenoph_Annot_Dpolygonoides.csv'
    images_folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_polygonoides/Censored__Annotated_Dioscorea_polygonoides/Archival_Components_Censored'
    output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3'
    classify_and_copy_images(csv_path_training_v_3, csv_path, images_folder_path, output_folder_path)

    csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_villosa-US/Phenoph_Annot_Dioscorea_villosa.csv'
    images_folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea_villosa-US/Censored__Annotated_Dioscorea_villosa/Archival_Components_Censored'
    output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3'
    classify_and_copy_images(csv_path_training_v_3, csv_path, images_folder_path, output_folder_path)

    csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-Colombia/Phenoph_Annot_DioscoreaColombia.csv'
    images_folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-Colombia/Censored__Annotated_DioscoreaColombia/Archival_Components_Censored'
    output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3'
    classify_and_copy_images(csv_path_training_v_3, csv_path, images_folder_path, output_folder_path)

    csv_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-HNC-ColombianHerbarium/Phenoph_Annot_DioscoreaHerbarium.csv'
    images_folder_path = 'C:/ML_Projects/Dan_Park/annotated_images/Dioscorea-HNC-ColombianHerbarium/Censored__Annotated_Dioscorea_HNCOL/Archival_Components_Censored'
    output_folder_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3'
    classify_and_copy_images(csv_path_training_v_3, csv_path, images_folder_path, output_folder_path)

    '''Save the dataset in the required format so you don't have to download from HF'''
    data_folders = {
        'train': 'D:/Dropbox/SwinV2_Classifier/data/training_v_3/train',
        'test': 'D:/Dropbox/SwinV2_Classifier/data/training_v_3/test'
    }
    class_names = ['both_both','female_both','female_flower','female_fruit','male_flower','sterile'] 
    dataset_path = 'D:/Dropbox/SwinV2_Classifier/data/training_v_3/dataset_dict'
    save_dataset_to_disk(dataset_path, data_folders, class_names)

    
