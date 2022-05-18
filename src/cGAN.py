
from torch.utils.data import Dataset, DataLoader
import glob
import random
import os
import numpy as np
from PIL import Image
from load_data import GANImageDataset


PATH = "../data/selfie2anime/"
# Find the dataset here: https://github.com/taki0112/UGATIT ...
# ... or directly under: https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing
BATCH_SIZE = 1


### define GAN model ###



if __name__ == '__main__':


    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    # parser.add_argument ... continue

    args = parser.parse_args()

    is_cuda_available = torch.cuda.is_available()
    print("Is CUDA available? {}".format(is_cuda_available))
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## the following might be needed for reproducibility:
    random.seed(5719)
    np.random.seed(5719)
    torch.manual_seed(5719)
    torch.use_deterministic_algorithms(True)

    # Read dataset: 

    training_dataset = ImageData(TRAIN_PATH, train=True)
    dev_dataset = ImageData(TEST_PATH, train=False)

    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


