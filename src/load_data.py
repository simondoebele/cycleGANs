import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class GANImageDataset(Dataset):
    def __init__(self, root_Images_A, root_Images_B):
        self.root_Images_A = root_Images_A
        self.root_Images_B = root_Images_B
        self.transform = A.Compose([
                                A.Resize(width=256, height=256),
                                A.HorizontalFlip(p=0.5),
                                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                                ToTensorV2()])

        self.A_images = os.listdir(root_Images_A)
        self.B_images = os.listdir(root_Images_B)

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
            A_img = self.transform(image = A_img)["image"]
            B_img = self.transform(image = B_img)["image"]

        return A_img, B_img



