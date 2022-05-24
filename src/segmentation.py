import torch
from PIL import Image
from torchvision import transforms
import os
from load_data import pil_loader
from device import DEVICE
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

path = "images4segmentation"


def load_images(path, images):
    original = []
    stylized = []
    images.sort()
    for img in images:
        img_path = os.path.join(path, img)
        if "X_real" in img:
            loaded = pil_loader(img_path)
            original.append(loaded)
        elif "Y_fake" in img:
            loaded = pil_loader(img_path)
            stylized.append(loaded)

    return original, stylized


def transform(image, trans, device):
    transformed = trans(image)
    input_batch = transformed.unsqueeze(0)
    input_batch = input_batch.to(device)
    return input_batch


images = os.listdir(path)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

to_tensor = transforms.ToTensor()

original, stylized = load_images(path, images)
segm_model = torch.hub.load('pytorch/vision:v0.10.0', 
                            'deeplabv3_resnet50', pretrained=True).to(DEVICE)
segm_model.eval()


for i, image in enumerate(original):
    output = segm_model(transform(image, preprocess, DEVICE))['out'][0]
    output_pred = output.argmax(0).unsqueeze(0).cpu().numpy()
    output_pred = np.where(output_pred != 15, 0, output_pred)
    mask = np.where(output_pred == 15, 1, output_pred).reshape((256, 256))
    mask = np.stack([mask, mask, mask])

    masked_stylized = mask * to_tensor(stylized[i]).numpy().reshape((3, 256, 256))
    combined = (1 - mask) * to_tensor(image).numpy().reshape((3, 256, 256)) + masked_stylized
    combined = torch.Tensor(combined)

    save_image(combined, f"combined/{i}.png")


