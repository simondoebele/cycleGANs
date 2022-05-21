from imageio import save
import torch
from load_data import GANImageDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


LR = 1e-5  # Learning rate
BETA1 = 0.5  # Beta 1 for Adam
BETA2 = 0.999  # Beta 2 for Adam
BSZ = 2  # Batch size
NUM_WRKS = 1  # Number of workers for dataloaders
EPOCHS = 10  # Number of epochs for training
L_CYCLE = 10  # Cycle loss weight
L_IDENT = 0.1  # Identity loss 
L_WD_DISC = 0.0001  # Weight decay for discriminator
L_WD_GEN = 0.0001  # Weight decay for generator
LOAD = False  # To load saved model
SAVE = True  # To save trained model at checkpoint
ID_LOSS = False  # If True, use id loss
W_REG = False  # If True, use L2 regularization
LR_SCH = False  # If True, use LR scheduling

TRAIN_DIR = "data/train/"  # Training directory
SAVED_GEN_X = "genx.pth.tar"
SAVED_GEN_Y = "geny.pth.tar"
SAVED_DISC_X = "discx.pth.tar"
SAVED_DISC_Y = "discy.pth.tar"
SAVED_IMG = "../saved_images/"
STATS_DIR = "../states/"
DATA_X = "horses"
DATA_Y = "zebras"



def train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, L1, MSE, 
             scaler_disc, scaler_gen, epoch, id_loss=False, weight_reg=False, 
             lr_sched=None):
    
    pbar = tqdm(desc=f"Epoch {epoch}", total=len(loader))
    for idx, (X, Y) in enumerate(loader):
        total_loss = 0.0

        with torch.cuda.amp.autocast():
            opt_disc.zero_grad()
        
            fake_X = gen_X(Y)
            d_X_real = disc_X(X)
            d_X_fake = disc_X(fake_X.detach())
            D_X_real_loss = MSE(d_X_real, torch.ones_like(d_X_real))
            D_X_fake_loss = MSE(d_X_fake, torch.ones_like(d_X_fake))
            D_X_Loss = D_X_fake_loss + D_X_real_loss

            fake_Y = gen_Y(X)
            d_Y_real = disc_Y(Y)
            d_Y_fake = disc_Y(fake_Y.detach())
            D_Y_real_loss = MSE(d_Y_real, torch.ones_like(d_Y_real))
            D_Y_fake_loss = MSE(d_Y_fake, torch.ones_like(d_Y_fake))
            D_Y_Loss = D_Y_fake_loss + D_Y_real_loss

            Disc_loss = D_X_Loss + D_Y_Loss
            if weight_reg:
                pass

            scaler_disc.scale(Disc_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()

        with torch.cuda.amp.autocast():
            opt_gen.zero_grad()

            # Adversarial loss
            d_X_fake = disc_X(fake_X)
            d_Y_fake = disc_Y(fake_Y)
            G_X_loss = MSE(d_X_fake, torch.ones_like(d_X_fake))
            G_Y_loss = MSE(d_Y_fake, torch.ones_like(d_Y_fake))
            G_loss = G_X_loss + G_Y_loss

            # Cycle loss
            cycle_X = gen_X(fake_Y)
            cycle_Y = gen_Y(fake_X)
            cycle_loss = L1(X, cycle_X) + L1(Y, cycle_Y)

            # Identity loss
            if id_loss:
                identity_X = gen_X(X)
                identity_Y = gen_Y(Y)
                identity_loss = L1(X, identity_X) + L1(Y, identity_Y)

            Gen_loss = G_loss + L_CYCLE * cycle_loss
            if id_loss:
                Gen_loss += L_IDENT * identity_loss
            if weight_reg:
                pass

            scaler_gen.scale(Gen_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()


        total_loss = Gen_loss.detach() + Disc_loss.detach()
        # TODO: Save losses to stats every 10 ind

        # Save images to saved_images every 200
        if idx % 200 == 0:
            save_image(fake_X*0.5 + 0.5, f"{SAVED_IMG}X_fake_{idx}.png")
            save_image(fake_Y*0.5 + 0.5, f"{SAVED_IMG}Y_fake_{idx}.png")
            save_image(X, f"{SAVED_IMG}X_real_{idx}.png")
            save_image(Y, f"{SAVED_IMG}X_real_{idx}.png")
        pbar.update(1)

    pbar.close()


def main(load_model=False, save_model=True):
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

    # loading checkpoint
    if load_model:
        load_checkpoint(SAVED_GEN_X, gen_X, gen_optimizer, LR)
        load_checkpoint(SAVED_GEN_Y, gen_Y, gen_optimizer, LR)
        load_checkpoint(SAVED_DISC_X, disc_X, disc_optimizer, LR)
        load_checkpoint(SAVED_DISC_Y, disc_Y, disc_optimizer, LR)


    # Create dataset
    dataset = GANImageDataset(TRAIN_DIR + DATA_X, TRAIN_DIR + DATA_Y)

    loader = DataLoader(
        dataset,
        batch_size=BSZ,
        shuffle=True,
        num_workers=NUM_WRKS,
        pin_memory=True
    )

    # Scalers
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # LR Scheduling
    lr_scheduler = None

    for epoch in range(EPOCHS):
        train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, 
                 disc_optimizer, gen_optimizer, L1_loss, MSE_loss, 
                 d_scaler, g_scaler, epoch, id_loss=ID_LOSS, weight_reg=W_REG,
                 lr_sched=lr_scheduler)

        # Save checkpoint
        if save_model:
            save_checkpoint(gen_X, gen_optimizer, filename=SAVED_GEN_X)
            save_checkpoint(gen_Y, gen_optimizer, filename=SAVED_GEN_Y)
            save_checkpoint(disc_X, disc_optimizer, filename=SAVED_DISC_X)
            save_checkpoint(disc_Y, disc_optimizer, filename=SAVED_DISC_Y)


main(load_model=LOAD, save_model=SAVE)
# TODO: Weight regularization
# TODO: LR scheduling
# TODO: Delete unnecessary files
# TODO: Save losses to stats every 10 ind





