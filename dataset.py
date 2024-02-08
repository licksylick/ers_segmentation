import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch

from utils import read_image, preprocess_image, reindex_mask


class SegmentationMulticlassDataset(Dataset):
    def __init__(self, df_path, is_train=False, augs=None, h_w=512, num_classes=2):
        self.df = pd.read_csv(df_path)

        self.augs = augs

        self.labels = [i for i in range(num_classes)]

        self.h_w = h_w

        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def mosaic(self, df):
        sub_df = df.sample(n=4, replace=True)

        img_path = [row['image_path'] for i, row in sub_df.iterrows()]
        msk_path = [row['mask_path'] for i, row in sub_df.iterrows()]

        half_h = self.h_w // 2
        half_w = self.h_w // 2

        r_image = np.zeros((self.h_w, self.h_w, 3), dtype=np.uint8)
        r_mask = np.zeros((self.h_w, self.h_w), dtype=np.uint8)

        for i, (img_p, msk_p) in enumerate(zip(img_path, msk_path)):

            image = cv2.imread(img_p)
            mask = cv2.imread(msk_p)

            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image, (self.h_w, self.h_w), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_gray, (self.h_w, self.h_w), interpolation=cv2.INTER_LINEAR)

            if i == 0:
                r_image[0:half_h, 0:half_w] = image[0:half_h, 0:half_w].copy()
                r_mask[0:half_h, 0:half_w] = mask[0:half_h, 0:half_w].copy()
            elif i == 1:
                r_image[0:half_h, half_w:] = image[0:half_h, half_w:].copy()
                r_mask[0:half_h, half_w:] = mask[0:half_h, half_w:].copy()
            elif i == 2:
                r_image[half_h:, 0:half_w] = image[half_h:, 0:half_w].copy()
                r_mask[half_h:, 0:half_w] = mask[half_h:, 0:half_w].copy()
            elif i == 3:
                r_image[half_h:, half_w:] = image[half_h:, half_w:].copy()
                r_mask[half_h:, half_w:] = mask[half_h:, half_w:].copy()

        return r_image, r_mask

    def __getitem__(self, index):
        img_path = self.df['image_path'][index]
        msk_path = self.df['mask_path'][index]

        image = read_image(img_path)
        mask = read_image(msk_path, to_rgb=False, flag=cv2.IMREAD_GRAYSCALE)

        if self.augs is not None:
            item = self.augs(image=image, mask=mask)
            image = item['image']
            mask = item['mask']

        if self.is_train:
            if np.random.random() < 0.25:
                image, mask = self.mosaic(self.df)

        image = preprocess_image(image, img_w=self.h_w, img_h=self.h_w)
        mask = cv2.resize(mask, (self.h_w, self.h_w), interpolation=cv2.INTER_NEAREST)
        sg_mask = torch.from_numpy(reindex_mask(mask, self.labels).copy())

        return {
            'image': image,
            'sg_mask': sg_mask.long(),
        }
