#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from generator import Generator
from PIL import Image
from torchvision.utils import make_grid
import io

default_model_path = "pix2pix/model/generator.pth"

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

def predict(src_image, model_path, image_size=(512, 512)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.nn.DataParallel(Generator().to(device))
    generator.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    generator.eval()
    pytorch_total_params = sum(p.numel() for p in generator.parameters())
    print(pytorch_total_params)

    src_img = src_image.convert("RGB")
    src_image = src_img.resize(image_size, Image.BILINEAR)
    src_image = (
        (torch.from_numpy(np.array(src_image)).permute(2, 0, 1).float()) - 127.5
    ) / 127.5
    inp = src_image.unsqueeze(0).to(device)
    out = generator(inp)
    image_tensor = (out + 1) / 2.0  # Denormalize image
    image_unflat = image_tensor.detach().cpu()  # convert gpu to cpu
    image_grid = make_grid(image_unflat[:1], nrow=1)  # using as make grid function
    img = image_grid.permute(1, 2, 0).squeeze()
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fronts from model")
    parser.add_argument("--model", default=default_model_path, help="Model path")
    parser.add_argument("source", type=argparse.FileType('rb'), help="Predict from given NWP model image")
    parser.add_argument("--save", type=argparse.FileType('w'), default=None, help="Save prediction as")
    args = parser.parse_args()

    print(args.source)
    src_image = Image.open(args.source)
    img = predict(src_image, args.model)
    plt.imshow(img)
    plt.show()