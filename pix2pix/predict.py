#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch
from generator import Generator
from PIL import Image
from torchvision.utils import make_grid

src_image_path = "datasets/generated_model/data/01_01_2021.png"
model_path = "pix2pix/model/generator.pth"
image_size = (512, 512)
device = "cuda" if torch.cuda.is_available() else "cpu"


def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32)):
    image_tensor = (image_tensor + 1) / 2.0  # Denormalize image
    image_unflat = image_tensor.detach().cpu()  # convert gpu to cpu
    image_grid = make_grid(
        image_unflat[:num_images], nrow=1
    )  # using as make grid function
    plt.imshow(
        image_grid.permute(1, 2, 0).squeeze()
    )  # convert image to right shape for require matplotlib
    plt.show()


generator = torch.nn.DataParallel(Generator().to(device))
generator.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
generator.eval()

src_img = Image.open(src_image_path).convert("RGB")
src_image = src_img.resize(image_size, Image.BILINEAR)
src_image = (
    (torch.from_numpy(np.array(src_image)).permute(2, 0, 1).float()) - 127.5
) / 127.5
inp = src_image.unsqueeze(0).to(device)
out = generator(inp)
image_tensor = (out + 1) / 2.0  # Denormalize image
image_unflat = image_tensor.detach().cpu()  # convert gpu to cpu
image_grid = make_grid(image_unflat[:1], nrow=1)  # using as make grid function
plt.imshow(image_grid.permute(1, 2, 0).squeeze())
plt.show()
