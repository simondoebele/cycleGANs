import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from random import shuffle, seed


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class GANImageDataset(Dataset):
    def __init__(self, root_Images_A, root_Images_B, portion=1.0):
        self.root_Images_A = root_Images_A
        self.root_Images_B = root_Images_B
        self.transform = A.Compose([
                                A.Resize(width=256, height=256),
                                A.HorizontalFlip(p=0.5),
                                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                                ToTensorV2()],
                                additional_targets={"image0": "image"}
                                )

        self.A_images = os.listdir(root_Images_A)
        self.B_images = os.listdir(root_Images_B)
        
        seed(42)
        shuffle(self.A_images)
        shuffle(self.B_images)

        self.A_images = self.A_images[:int(portion * len(self.A_images))]
        self.B_images = self.B_images[:int(portion * len(self.B_images))]

        self.length_dataset = max(len(self.A_images), len(self.B_images)) # 1000, 1500
        self.A_num_images = len(self.A_images)
        self.B_num_images = len(self.B_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # do % here as we might not have same dataset sizes (i.e. matching A and B images)
        A_img = self.A_images[index % self.A_num_images]
        B_img = self.B_images[index % self.B_num_images]

        A_path = os.path.join(self.root_Images_A, A_img)
        B_path = os.path.join(self.root_Images_B, B_img)

        A_img = np.array(pil_loader(A_path))
        B_img = np.array(pil_loader(B_path))

        if self.transform:
            transformed = self.transform(image = A_img, image0=B_img)
            A_img = transformed["image"]
            B_img = transformed["image0"]
        
        return A_img, B_img



