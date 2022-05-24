import torch
from load_data import GANImageDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
import csv
from device import DEVICE
import os
from torch.optim.lr_scheduler import LinearLR, CyclicLR


LR = 1e-4  # Learning rate
BETA1 = 0.5  # Beta 1 for Adam
BETA2 = 0.999  # Beta 2 for Adam
BSZ = 1  # Batch size
NUM_WRKS = 4  # Number of workers for dataloaders
EPOCHS = 100  # Number of epochs for training
L_CYCLE = 10  # Cycle loss weight
L_IDENT = 0.1  # Identity loss
USE_WEIGHT_DECAY = False  # use L2 regularization
L_WD_DISC = 0.0001  # Weight decay for discriminator
L_WD_GEN = 0.0001  # Weight decay for generator
LOAD = False  # To load saved model
SAVE = True  # To save trained model at checkpoint
ID_LOSS = False  # If True, use id loss
W_REG = False  # If True, use L2 regularization
LR_SCH_LIN = False  # If True, use LR scheduling with linear decay
# start factor for LR_SCH_STEP (initial LR is multiplied by this)
LR_SCH_START_FACTOR = 0.5
LR_SCH_CYC = True  # If True, use LR scheduling with cyclic LR
LR_MIN_CYC = 1e-5  # min for cyclic LR
LR_MAX_CYC = 1e-1  # max for cyclic LR

TRAIN_DIR = "data/train/"  # Training directory
TEST_DIR = "data/test/"  # Test directory
SAVED_IMG = "saved_images/"
STATS_DIR = "stats/"

# REMEMBER TO CHANGE THESE BETWEEN EXPERIMENTS
RUN_NAME = "anime_optimizer"
SAVED_GEN_X = f"models/genx_{RUN_NAME}.pth.tar"
SAVED_GEN_Y = f"models/geny_{RUN_NAME}.pth.tar"
SAVED_DISC_X = f"models/discx_{RUN_NAME}.pth.tar"
SAVED_DISC_Y = f"models/discy_{RUN_NAME}.pth.tar"
# REMEMBER TO CHANGE THESE BETWEEN EXPERIMENTS
DATA_X = "selfies"
DATA_Y = "anime"

try:
    os.mkdir(f"{SAVED_IMG}{RUN_NAME}/")
except FileExistsError:
    pass
try:
    os.mkdir(f"{SAVED_IMG}{RUN_NAME}/train")
except FileExistsError:
    pass
try:
    os.mkdir(f"{SAVED_IMG}{RUN_NAME}/test")
except FileExistsError:
    pass


def train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, L1, MSE,
             scaler_disc, scaler_gen, epoch, id_loss=False, weight_reg=False,
             lr_scheduler_disc=None, lr_scheduler_gen=None):

    pbar = tqdm(desc=f"Epoch {epoch}", total=len(loader))
    X_reals = 0
    X_fakes = 0

    for idx, (X, Y) in enumerate(loader):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        identity_loss = torch.tensor(0.0)
        Disc_weight_loss = torch.tensor(0.0)
        Gen_weight_loss = torch.tensor(0.0)

        with torch.cuda.amp.autocast():
            fake_X = gen_X(Y)
            d_X_real = disc_X(X)
            d_X_fake = disc_X(fake_X.detach())
            X_reals += d_X_real.mean().item()
            X_fakes += d_X_fake.mean().item()
            D_X_real_loss = MSE(d_X_real, torch.ones_like(d_X_real))
            D_X_fake_loss = MSE(d_X_fake, torch.zeros_like(d_X_fake))
            D_X_Loss = D_X_fake_loss + D_X_real_loss

            fake_Y = gen_Y(X)
            d_Y_real = disc_Y(Y)
            d_Y_fake = disc_Y(fake_Y.detach())
            D_Y_real_loss = MSE(d_Y_real, torch.ones_like(d_Y_real))
            D_Y_fake_loss = MSE(d_Y_fake, torch.zeros_like(d_Y_fake))
            D_Y_Loss = D_Y_fake_loss + D_Y_real_loss

            Disc_loss = (D_X_Loss + D_Y_Loss)/2
            if weight_reg:
                pass

            opt_disc.zero_grad()
            scaler_disc.scale(Disc_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()

        with torch.cuda.amp.autocast():
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

            opt_gen.zero_grad()
            scaler_gen.scale(Gen_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

        total_loss = Gen_loss.detach() + Disc_loss.detach()
        # Save losses to stats every 10 ind
        if idx % 10 == 0:
            try:
                open(f"{STATS_DIR}{RUN_NAME}.csv", "r")
                f = open(f"{STATS_DIR}{RUN_NAME}.csv", "a")
                writer = csv.writer(f)
                csv_line = [total_loss.item(), Disc_loss.item(), G_loss.item(), cycle_loss.item(),
                            identity_loss.item(), Disc_weight_loss.item(), Gen_weight_loss.item()]
                writer.writerow(csv_line)
            except FileNotFoundError:
                f = open(f"{STATS_DIR}{RUN_NAME}.csv", "w")
                writer = csv.writer(f)
                head_line = ["total_loss", "D_loss", "G_loss", "cycle_loss",
                             "identity_loss", "Disc_weight_loss", "Gen_weight_loss"]
                csv_line = [total_loss.item(), Disc_loss.item(), G_loss.item(), cycle_loss.item(),
                            identity_loss.item(), Disc_weight_loss.item(), Gen_weight_loss.item()]
                writer.writerow(head_line)
                writer.writerow(csv_line)
            f.close()
            # Save total_loss, gen_loss, disc_loss, weight_reg_cost, etc.

        # Save images to saved_images every 200
        if idx % 400 == 0:
            save_image(fake_X*0.5 + 0.5,
                       f"{SAVED_IMG}{RUN_NAME}/train/{epoch}_{idx}_X_fake.png")
            save_image(fake_Y*0.5 + 0.5,
                       f"{SAVED_IMG}{RUN_NAME}/train/{epoch}_{idx}_Y_fake.png")
            save_image(
                X, f"{SAVED_IMG}{RUN_NAME}/train/{epoch}_{idx}_X_real.png")
            save_image(
                Y, f"{SAVED_IMG}{RUN_NAME}/train/{epoch}_{idx}_Y_real.png")
        pbar.update(1)

    pbar.close()
    if lr_scheduler_disc and lr_scheduler_gen:
        lr_scheduler_disc.step()
        lr_scheduler_gen.step()


def test(gen_X, gen_Y, loader):
    pbar = tqdm(desc=f"Test", total=len(loader))

    for idx, (X, Y) in enumerate(loader):
        with torch.cuda.amp.autocast():
            fake_X = gen_X(Y)
            fake_Y = gen_Y(X)

        # Save images to saved_images
        save_image(fake_X*0.5 + 0.5,
                   f"{SAVED_IMG}{RUN_NAME}/test/{idx}_X_fake.png")
        save_image(fake_Y*0.5 + 0.5,
                   f"{SAVED_IMG}{RUN_NAME}/test/{idx}_Y_fake.png")
        save_image(X, f"{SAVED_IMG}{RUN_NAME}/test/{idx}_X_real.png")
        save_image(Y, f"{SAVED_IMG}{RUN_NAME}/test/{idx}_Y_real.png")
        pbar.update(1)

    pbar.close()


def main(load_model=False, save_model=True):
    # check correct LR scheduling:
    assert ((LR_SCH_LIN or LR_SCH_CYC) or (not LR_SCH_LIN and not LR_SCH_CYC))

    disc_X = Discriminator(in_channels=3)
    disc_Y = Discriminator(in_channels=3)
    gen_X = Generator(img_channels=3, num_residuals=9)
    gen_Y = Generator(img_channels=3, num_residuals=9)

    weight_decay_disc = 0
    weight_decay_gen = 0
    if USE_WEIGHT_DECAY:
        weight_decay_disc = L_WD_DISC
        weight_decay_gen = L_WD_GEN

    # disc_optimizer = optim.Adam(
    #     list(disc_X.parameters()) + list(disc_Y.parameters()),
    #     lr=LR,
    #     betas=(BETA1, BETA2),
    #     weight_decay=weight_decay_disc
    # )

    # gen_optimizer = optim.Adam(
    #     list(gen_X.parameters()) + list(gen_Y.parameters()),
    #     lr=LR,
    #     betas=(BETA1, BETA2),
    #     weight_decay=weight_decay_gen
    # )

    disc_optimizer = optim.SGD(
        list(disc_X.parameters()) + list(disc_Y.parameters()), lr=LR, momentum=0.9)

    gen_optimizer = optim.SGD(
        list(gen_X.parameters()) + list(gen_Y.parameters()), lr=LR, momentum=0.9)

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    # loading checkpoint
    if load_model:
        load_checkpoint(SAVED_GEN_X, gen_X, gen_optimizer, LR)
        load_checkpoint(SAVED_GEN_Y, gen_Y, gen_optimizer, LR)
        load_checkpoint(SAVED_DISC_X, disc_X, disc_optimizer, LR)
        load_checkpoint(SAVED_DISC_Y, disc_Y, disc_optimizer, LR)

    # Create dataset
    if DATA_X == "selfies":
        portion = 0.5
    else:
        portion = 1.0
    dataset = GANImageDataset(
        TRAIN_DIR + DATA_X, TRAIN_DIR + DATA_Y, portion=portion)

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
    if LR_SCH_LIN:
        # total_iters=after how many iters to stop the decay
        lr_scheduler_disc = LinearLR(
            disc_optimizer, start_factor=LR_SCH_START_FACTOR, total_iters=EPOCHS//2)
        lr_scheduler_gen = LinearLR(
            gen_optimizer, start_factor=LR_SCH_START_FACTOR, total_iters=EPOCHS//2)
    if LR_SCH_CYC:
        lr_scheduler_disc = CyclicLR(
            disc_optimizer, base_lr=LR_MIN_CYC, max_lr=LR_MAX_CYC, cycle_momentum=False)
        lr_scheduler_gen = CyclicLR(
            gen_optimizer, base_lr=LR_MIN_CYC, max_lr=LR_MAX_CYC, cycle_momentum=False)

    for epoch in range(EPOCHS):
        train_fn(disc_X, disc_Y, gen_X, gen_Y, loader,
                 disc_optimizer, gen_optimizer, L1_loss, MSE_loss,
                 d_scaler, g_scaler, epoch, id_loss=ID_LOSS, weight_reg=W_REG,
                 lr_scheduler_disc=lr_scheduler_disc, lr_scheduler_gen=lr_scheduler_gen)

        # Save checkpoint
        if save_model:
            save_checkpoint(gen_X, gen_optimizer, filename=SAVED_GEN_X)
            save_checkpoint(gen_Y, gen_optimizer, filename=SAVED_GEN_Y)
            save_checkpoint(disc_X, disc_optimizer, filename=SAVED_DISC_X)
            save_checkpoint(disc_Y, disc_optimizer, filename=SAVED_DISC_Y)

    print("Training complete!")
    # Create dataset
    dataset = GANImageDataset(TEST_DIR + DATA_X, TEST_DIR + DATA_Y)

    loader = DataLoader(
        dataset,
        batch_size=BSZ,
        shuffle=True,
        num_workers=NUM_WRKS,
        pin_memory=True
    )

    test(gen_X, gen_Y, loader)


main(load_model=LOAD, save_model=SAVE)
# TODO: Weight regularization, remember to normalize by the number of parameters of generator and discriminator
# TODO: Use a different optimizer than Adam
# TODO: Add different dataset under data/train and data/test
# TODO: Implement extension with mask
