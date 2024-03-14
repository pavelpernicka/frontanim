#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from discriminator import Discriminator
from generator import Generator
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

target_folder = "pix2pix/model"
fronts_path = "datasets/generated_model/fronts"  # ground truth
data_path = "datasets/generated_model/data"  # nwp model based data
fronts_path_list = []

for filename in os.listdir(fronts_path):
    potential_second = os.path.join(data_path, filename)
    if os.path.isfile(potential_second):
        fronts_path_list.append(filename)
    else:
        print(f"Excluding {filename} because it cannot be paired")

print(fronts_path)
print(f"Total images: {len(fronts_path)}")


class CustomDataset(Dataset):
    def __init__(
        self,
        path_list,
        src_path,
        dst_path,
        image_size=(512, 512),  # image_size must be square
    ):
        self.path_list = path_list
        self.src_path = src_path
        self.dst_path = dst_path
        self.image_size = image_size

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        """
        return meteodata and fronts tensors at index idx
        """
        src_image_path = os.path.join(self.src_path, self.path_list[idx])
        dst_image_path = os.path.join(self.dst_path, self.path_list[idx])

        src_img = Image.open(src_image_path).convert("RGB")
        dst_img = Image.open(dst_image_path).convert("RGB")

        src_image = src_img.resize(self.image_size, Image.BILINEAR)
        dst_image = dst_img.resize(self.image_size, Image.BILINEAR)

        src_image = (
            (torch.from_numpy(np.array(src_image)).permute(2, 0, 1).float())
            - 127.5  # normalize and create tensor
        ) / 127.5
        dst_image = (
            (torch.from_numpy(np.array(dst_image)).permute(2, 0, 1).float()) - 127.5
        ) / 127.5

        return src_image, dst_image


def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32), fn="result.png"):
    plt.clf()
    image_tensor = (image_tensor + 1) / 2.0  # denormalize
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(fn)


def mk_loss_chart(d_losses, g_losses, gm_losses, fn="loss.png"):
    plt.clf()
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(gm_losses, label="L1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(fn)


train_dataset = CustomDataset(fronts_path_list, data_path, fronts_path)
print(f"Main Dataset Length: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8)

# discriminator model
discriminator = torch.nn.DataParallel(Discriminator().to(device))
print(discriminator)

# generator model
generator = torch.nn.DataParallel(Generator().to(device))
print(generator)

# training
epoch = 350
L1_lambda = 0.25  # how much does difference between generated and target matter (bast values are 2-3)
# Adam parameters
learning_rate = 0.0001
beta_value = (
    0.5,
    0.999,
)
gen_opt = Adam(
    generator.parameters(), lr=learning_rate, betas=beta_value
)  # Generator Optimizer
disc_opt = Adam(
    discriminator.parameters(), lr=learning_rate, betas=beta_value
)  # Discriminator Optimizer
bc_loss = nn.BCELoss()
m_loss = nn.L1Loss()

d_losses = []
g_losses = []
gm_losses = []

# write config for future use
f = open(f"{target_folder}/config.txt", "w")
f.write(
    f"epoch: {epoch}\nL1_lambda: {L1_lambda}\nlearning_rate: {learning_rate}\nbeta_value: {beta_value}"
)
f.close()

print("Training...")
for e in range(epoch):  # train whole dataset epoch times
    print(f"Epoch: {e}")
    for src_images, dst_images in tqdm(train_loader):
        # Move tensors to specefic device
        src_images = src_images.to(device)
        dst_images = dst_images.to(device)

        # Discriminator
        # Real
        discriminator.zero_grad()  # Reset the gradient of the model parameter
        real_pred = discriminator(
            src_images, dst_images
        )  # First discriminator real data see.
        rb_loss = bc_loss(
            real_pred, torch.ones_like(real_pred)
        )  # Compute the binary cross entropy loss between the discriminator's real predictions and a tensor of ones.

        # Fake Train
        fake_sample = generator(src_images)
        fake_pred = discriminator(
            src_images, fake_sample.detach()
        )  # Discriminator now to see fake data and also .detach() method used to remove from the computation graph of the discriminator.
        fb_loss = bc_loss(
            fake_pred, torch.zeros_like(fake_pred)
        )  # Now compute the binary-cross entropy loss between the discriminator's fake prediction and a tensor of zeros.

        # Combine real loss and fake loss
        d_loss = rb_loss + fb_loss
        d_loss.backward()  # Backpropagate the discriminator's loss through the model.
        disc_opt.step()  # Update the parameters of the discriminator model using the Adam optimizer.

        # Generator
        gen_opt.zero_grad()  # Rest the Generator model parameter similar to discriminator
        fake_pred2 = discriminator(
            src_images, fake_sample
        )  # Discriminator takes an src and generates images and returns a prediction if it's real or not.
        gb_loss = bc_loss(
            fake_pred2, torch.ones_like(fake_pred2)
        )  # Compute the binary-cross entropy loss between discriminator fake prediction and tensor of ones.
        gm_loss = m_loss(
            fake_sample, dst_images
        )  # And also calculate L1 loss for the model to see the difference between generated and target image.

        g_loss = (
            gb_loss + L1_lambda * gm_loss
        )  # Combine these two losses and l1_lambda add to control the weight of the L1 loss.
        g_loss.backward()  # Backpropagate the generator loss through the model
        gen_opt.step()  # Update the generator parameter for using Adam optimizer.
    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    gm_losses.append(gm_loss.item())

    show_tensor_images(src_images, num_images=1, fn=f"{target_folder}/src_{e}.png")
    show_tensor_images(dst_images, num_images=1, fn=f"{target_folder}/dst_{e}.png")
    show_tensor_images(fake_sample, num_images=1, fn=f"{target_folder}/gen_{e}.png")
    mk_loss_chart(d_losses, g_losses, gm_losses, fn=f"{target_folder}/loss_chart.png")

    torch.save(generator.state_dict(), f"{target_folder}/generator.pth")
    torch.save(discriminator.state_dict(), f"{target_folder}/discriminator.pth")
