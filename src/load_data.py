
from torch.utils.data import Dataset, DataLoader
import glob
import random
import os
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class GANImageDataset(Dataset):
    def __init__(self, root, train=True):
        if train:
        	# we might be interested in transforming our images. change this as we see fit.
        	# see: https://pytorch.org/vision/main/transforms.html
            self.transform = trans.Compose([trans.RandomHorizontalFlip(),
                                            trans.Resize((int(CONFIG.size_h * 0.5), int(CONFIG.size_w * 0.5)), Image.BICUBIC),
                                            trans.RandomCrop((CONFIG.size_h, CONFIG.size_w)),
                                            trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.A_file = sorted(glob.glob(os.path.join(root, 'trainA') + '/*.jpg'))
            self.B_file = sorted(glob.glob(os.path.join(root, 'trainB') + '/*.jpg'))
        else:
            self.transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.A_file = sorted(glob.glob(os.path.join(root, 'testA') + '/*.jpg'))
            self.B_file = sorted(glob.glob(os.path.join(root, 'testB') + '/*.jpg'))

    def __getitem__(self, index):
    	# get matching "real" (A) and "style" (B) images
    	A_path = self.A_file[index % len(self.A_file)]
    	B_path = self.B_file[random.randint(0, len(self.B_file) - 1)]
        A = self.transform(pil_loader(A_path))
        B = self.transform(pil_loader(B_path))

        return A, B

    def __len__(self):
        return max(len(self.A_file), len(self.B_file))