import os
import tarfile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision import transforms

root = './dataset/cub'
base_folder = 'CUB_200_2011/images'
file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
filename = 'CUB_200_2011.tgz'
tgz_md5 = '97eceeb196236b17998738112f37df78'

def name_to_class_names_cub_2011() -> tuple:
    class_names = []
    with open(os.path.join(root, 'CUB_200_2011', 'classes.txt'), "r") as file:
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                class_name = " ".join(parts[1].split(".", 1)[1].split("_"))
                class_names.append(class_name)
    return class_names

def load_cub_2011_train_test_as_numpy(image_size: int=256):
    download_cub_2011()

    base_folder = 'CUB_200_2011'
    images_file = os.path.join(root, base_folder, 'images.txt')
    labels_file = os.path.join(root, base_folder, 'image_class_labels.txt')
    split_file = os.path.join(root, base_folder, 'train_test_split.txt')
    images_folder = os.path.join(root, base_folder, 'images')

    images_df = pd.read_csv(images_file, sep=' ', names=['img_id', 'filepath'])
    labels_df = pd.read_csv(labels_file, sep=' ', names=['img_id', 'target'])
    split_df = pd.read_csv(split_file, sep=' ', names=['img_id', 'is_train'])

    images_df['class'] = images_df['filepath'].apply(lambda x: x.split('/')[0].split('.')[-1])
    images_df['suffix'] = images_df['class'].apply(lambda x: x.split('_')[-1])
    images_df['suffix_id'] = pd.factorize(images_df['suffix'])[0]

    data_df = images_df.merge(labels_df, on='img_id').merge(split_df, on='img_id')

    data_df['target'] = data_df['target'] - 1

    def load_images(filepaths):
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size)
        ])
        image_list = []
        for filepath in tqdm(filepaths, desc="Loading Images", leave=False, ncols=50):
            img_path = os.path.join(images_folder, filepath)
            img = Image.open(img_path).convert("RGB") 
            # img = img.resize((image_size,image_size))
            img = transform(img)
            img_array = np.array(img)
            image_list.append(img_array)
        return np.array(image_list)

    train_df = data_df[data_df['is_train'] == 1]
    test_df = data_df[data_df['is_train'] == 0]

    train_images = load_images(train_df['filepath'])
    train_labels_full = train_df['target'].values
    train_labels_weak = train_df['suffix_id'].values

    test_images = load_images(test_df['filepath'])
    test_labels_full = test_df['target'].values
    test_labels_weak = test_df['suffix_id'].values

    return train_images, train_labels_full, train_labels_weak, test_images, test_labels_full, test_labels_weak

def download_cub_2011():
    if check_cub_2011_integrity(root):
        print('Files already downloaded and verified')
        return

    download_file_from_google_drive(file_id, root, filename, tgz_md5)

    with tarfile.open(os.path.join(root, filename), "r:gz") as tar:
        tar.extractall(path=root)
        print(f"Extracted {filename} to {root}")
    
    print("Download and extraction completed.")

def check_cub_2011_integrity(root):
    dataset_folder = os.path.join(root, 'CUB_200_2011')
    return os.path.isdir(dataset_folder)

if __name__ == '__main__':
    train_images, train_labels_full, train_labels_weak, test_images, test_labels_full, test_labels_weak = load_cub_2011_train_test_as_numpy()
    print(train_images.shape, train_labels_full.shape, train_labels_weak.shape)
    print(test_images.shape, test_labels_full.shape, test_labels_weak.shape)