import cv2
import albumentations as A


class Augs:
    def __init__(self, h_w):
        self.h_w = h_w

    def train_augs(self):
        return A.Compose([
            A.Resize(self.h_w, self.h_w, p=1.0),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=35, p=0.5,
                               border_mode=cv2.BORDER_REFLECT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.5),
                A.JpegCompression(p=1.0, quality_lower=50, quality_upper=100)
            ], p=0.35),
            A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.07, 0.07), brightness_by_max=True)
        ], p=1.0)

    def val_test_augs(self):
        return A.Compose([
            A.Resize(self.h_w, self.h_w, p=1.0)
        ])
