import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import shutil
import torch
from collections import defaultdict
import importlib
from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


def copy_files(src_paths, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for src_path in src_paths:
        shutil.copy(src_path, dest_folder)



def save_config(config):
    os.makedirs(config['common']['save_path'], exist_ok=True)
    with open(f"{config['common']['save_path']}/config.yaml", 'w') as file:
        OmegaConf.save(config=config, f=file)


def preprocess_config(config):
    exp_name = config['common'].get('exp_name', 'exp0')
    config['common']['save_path'] = os.path.join(exp_name)

    # Overwrite some params
    max_epochs = config['common'].get('max_epochs', False)
    if max_epochs:
        config['trainer']['params']['max_epochs'] = max_epochs

    return config



def read_image(img_path: str, to_rgb: bool=True, flag: int=cv2.IMREAD_COLOR) -> np.array:
    '''
    img_path: path to image
    to_rgb: apply cv2.COLOR_BGR2RGB or not
    flag: [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED]
    '''
    image = cv2.imread(img_path, flag)
    if image is None:
        raise FileNotFoundError(f'{img_path}')
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def split_df(df):
    dfs = []

    for _, row in df.iterrows():
        mask_path = row['mask_path']
        images_path = row['images_path']
        # target = row['target']

        # images_path = ast.literal_eval(images_path)


        temp_df = pd.DataFrame(
            {'mask_path': [mask_path] * len(images_path), 'image_path': images_path})

        dfs.append(temp_df)

    new_df = pd.concat(dfs, ignore_index=True)
    return new_df

def prepare_stratified_train_val_test_csv(input_path, test_val_size=0.3, random_state=42):
    images_path = os.path.join(input_path, 'images')
    masks_path = os.path.join(input_path, 'annotations')

    mask_files = os.listdir(masks_path)

    im_masks_dict = defaultdict(list)

    for mask_file in mask_files:
        for i in range(6, 9):
            image_path = os.path.join(images_path, mask_file.replace('_mask.png', f'_{i}.png'))
            mask_path = os.path.join(masks_path, mask_file)

            mask = read_image(mask_path)
            classes = np.unique(mask)

            class_key = tuple(sorted(classes))
            im_masks_dict[class_key].append((image_path, mask_path))

    flat_data = [(image, mask, key) for key, value_list in im_masks_dict.items() for image, mask in value_list]

    df = pd.DataFrame(flat_data, columns=['image_path', 'mask_path', 'mask_classes'])

    df['mask_classes'] = df['mask_classes'].apply(
        lambda x: tuple(map(int, x.strip('()').split(','))) if isinstance(x, str) else x)

    result_df = df.groupby('mask_path').agg({'image_path': tuple, 'mask_classes': 'first'}).reset_index()
    result_df.columns = ['mask_path', 'images_path', 'mask_classes']

    df = result_df

    df['unique_mask_classes'] = pd.Categorical(df['mask_classes']).codes

    unique_mask_classes = df['unique_mask_classes'].unique()

    target_mapping = {classes: i for i, classes in enumerate(unique_mask_classes)}

    df['target'] = df['unique_mask_classes'].map(target_mapping)
    df = df.drop(columns=['unique_mask_classes'])
    df = df.reset_index(drop=True)

    # Filter out rows with targets having less than or equal to 2 occurrences
    df_filtered = df.groupby('target').filter(lambda x: len(x) > 5)

    df_train, df_temp = train_test_split(df_filtered, test_size=test_val_size, random_state=42, stratify=df_filtered['target'])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['target'])

    df_train.reset_index(drop=True)
    df_train = split_df(df_train)
    df_val.reset_index(drop=True)
    df_val = split_df(df_val)
    df_test.reset_index(drop=True)
    df_test = split_df(df_test)

    df_train.to_csv('train.csv', index=False)
    df_val.to_csv('val.csv', index=False)
    df_test.to_csv('test.csv', index=False)


def preprocess_image(image, img_w=None, img_h=None, interpolation=cv2.INTER_LINEAR, mean=np.array([0, 0, 0]),
                     std=np.array([1, 1, 1])):
    '''
    mean=[0., 0., 0.], std=[1., 1., 1.]
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    '''
    img = image.copy()
    if img_w and img_h:
        img = cv2.resize(img, (img_w, img_h), interpolation=interpolation)
    if img.ndim == 2:
        img = img[:, :, None]
    img = ((img.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def reindex_mask(image, labels):
    img = image.copy()
    for index, label in enumerate(labels):
        img[img == label] = index
    return img

