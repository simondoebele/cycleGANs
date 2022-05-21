import torch
# from load_data import GANImageDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# import config
# from utils import save_checkpoint, load_checkpoint
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


LR = 0.0001
BETA1 = 0.5
BETA2 = 0.999
TRAIN_DIR = ""
BSZ = 2
NUM_WRKS = 1
EPOCHS = 10


def train_fn(discriminators, generators, loader, optimizers, L1, MSE, scalers, epoch):
    # Remember which is disc1, gen1 and disc2, gen2
    disc_X, disc_Y = discriminators
    gen_X, gen_Y = generators
    opt_disc, opt_gen = optimizers
    scaler_disc, scaler_gen = scalers

    pbar = tqdm(desc=f"Epoch {epoch}", total=len(loader))
    for idx, (X, Y) in enumerate(loader):
        pbar.update(1)
    pbar.close()


def main():
    disc_X = Discriminator(in_channels=3)
    disc_Y = Discriminator(in_channels=3)
    gen_X = Generator(img_channels=3, num_residuals=9)
    gen_Y = Generator(img_channels=3, num_residuals=9)

    disc_optimizer = optim.Adam(
        list(disc_X.parameters()) + list(disc_Y.parameters()),
        lr=LR,
        betas=(BETA1, BETA2)
    )

    gen_optimizer = optim.Adam(
        list(gen_X.parameters()) + list(gen_Y.parameters()),
        lr=LR,
        betas=(BETA1, BETA2)
    )

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    # Add loading checkpoint

    # Create dataset, transforms (?)
    dataset = None

    loader = DataLoader(
        dataset,
        batch_size=BSZ,
        shuffle=True,
        num_workers=NUM_WRKS,
        pin_memory=True
    )

    # Scalers?
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        train_fn((disc_X, disc_Y), (gen_X, gen_Y), loader, 
                 (disc_optimizer, gen_optimizer), L1_loss, MSE_loss, 
                 (d_scaler, g_scaler), epoch)

        # Save checkpoint




