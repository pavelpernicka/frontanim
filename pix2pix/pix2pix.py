#!/usr/bin/env python3
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.autograd import Variable
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch import zeros, ones
from torch.optim import Adam
from torch.nn.init import normal_
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
device = 'cuda' if torch.cuda.is_available() else "cpu"
print("Using device: " + device)

#parameters
epoch = 350 # It's used for how many time the model train the entire dataset.
L1_weight = 1
Fronts_weight = 1.5
learning_rate = 0.0002 # It's helping for how much model parameter update during training.
beta_value = (0.5, 0.999) # It controls how much the optimizer remembers the previous gradient.
num_filters = 64 #parameter for Generator and Discriminator
image_size = 256 #target size of images in dataset

target_folder = "pix2pix/model"
fronts_path = "datasets/generated_model-v4/fronts" #ground truth
data_path = "datasets/generated_model-v4/data" #nwp model based data
fronts_path_list = []

for filename in os.listdir(fronts_path):
  potential_second = os.path.join(data_path, filename)
  if os.path.isfile(potential_second):
    fronts_path_list.append(filename);
  else:
    print(f"Excluding {filename} because it cannot be paired")
    
print(fronts_path)
print(f"Total images: {len(fronts_path_list)}")

class CustomDataset(Dataset):

  """
    CustomDataset class is a subclass of the Dataset class in PyTorch. This means that it can be used to create a dataset of images and sketches for training a machine learning model.

    Args:
        path_list (list of str): A list of paths to the image and sketch files.
        src_path (str): The path to the directory where the image files are located.
        dst_path (str): The path to the directory where the sketch files are located.
        ext (str): The file extension of the image and sketch files.
        image_size (int): The size of the images and sketches after they have been resized.

    Returns: RGB photo and sketch image.
  """
  def __init__(self, path_list, src_path, dst_path,ext='.png', image_size=(image_size, image_size)):
    self.path_list = path_list
    self.src_path = src_path
    self.dst_path = dst_path
    self.ext = ext
    self.image_size = image_size

  # This len method for return length of the dataset
  def __len__(self):
    return len(self.path_list)

  # This is the most important method in this class.
  def __getitem__(self, idx):
    """
    Gets the image and sketch image at a particular index from the dataset.

    Args:
      idx: The index of the image and sketch to get.

    Returns:
      A tuple of the image and sketch tensors.
    """
    # First join two path src folder and get path to the image file.
    src_image_path = os.path.join(self.src_path, self.path_list[idx])
    dst_image_path = os.path.join(self.dst_path, self.path_list[idx])

    # Load the image and sketch from the filesystem.
    src_img = Image.open(src_image_path).convert('RGB')
    dst_img = Image.open(dst_image_path).convert('RGB')
    
    # Resize the image and sketch to the specified size.
    src_image = src_img.resize(self.image_size,Image.BILINEAR)
    dst_image = dst_img.resize(self.image_size,Image.BILINEAR)

   # Convert image to tensors and also normalize them
    src_image = ((torch.from_numpy(np.array(src_image)).permute(2,0,1).float() ) - 127.5)/127.5
    dst_image = ((torch.from_numpy(np.array(dst_image)).permute(2,0,1).float())- 127.5)/127.5

    return src_image, dst_image
    
def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32), fn="result.png"):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    plt.clf()
    image_tensor = (image_tensor + 1) / 2.0 # Denormalize image
    image_unflat = image_tensor.detach().cpu() # convert gpu to cpu
    image_grid = make_grid(image_unflat[:num_images], nrow=1) # using as make grid function
    plt.imshow(image_grid.permute(1, 2, 0).squeeze()) # convert image to right shape for require matplotlib
    plt.savefig(fn) 
    
def mk_loss_chart(d_losses, g_losses, fn="loss.png"):
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Generator loss', color=color)
	ax1.plot(g_losses, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()

	color = 'tab:blue'
	ax2.set_ylabel('Discriminator loss', color=color)
	ax2.plot(d_losses, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	plt.title("Training losses")
	plt.savefig(fn)
	plt.close()
    
def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)
    
def hinge_loss(y_pred, target):
    loss = nn.ReLU()(1.0 - (y_pred * target)).mean()
    return loss
    
class CustomLoss(nn.Module):
    def __init__(self, l1_weight=1.0, line_weight=1):
        super(CustomLoss, self).__init__()
        self.l1_weight = l1_weight
        self.line_weight = line_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_images, dst_images):
        # Calculate L1 loss between generated and ground truth images
        l1_loss = self.l1_loss(gen_images, dst_images)
        
        # Extract line regions from ground truth images (assuming lines are represented as non-white pixels)
        line_mask = (dst_images.sum(dim=1) != 3.0).float()  # Sum RGB channels and check if != 3.0 (non-white)
        
        # Calculate line loss by comparing the difference between generated and ground truth pixel values
        line_loss = torch.abs(gen_images - dst_images)
        line_loss = line_loss * line_mask.unsqueeze(1)  # Apply mask to focus only on line regions
        line_loss = line_loss.sum() / line_mask.sum()  # Normalize by the number of non-white pixels
        
        # Combine L1 loss and line loss with weights
        total_loss = self.l1_weight * l1_loss + self.line_weight * line_loss
        
        return total_loss
        
# image_size=(600, 463)
train_dataset = CustomDataset(fronts_path_list, data_path, fronts_path)
print(f"Main Dataset Length: {len(train_dataset)}")

# If your dataset is ready. now time to move on and create a PyTorch DataLoader object.
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

# Model setup
discriminator = Discriminator(num_filter=num_filters).to(device)
generator = Generator(num_filter=num_filters).to(device)
generator.normal_weight_init(mean=0.0, std=0.02)
discriminator.normal_weight_init(mean=0.0, std=0.02)

# Optimizer setup
G_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=beta_value)
D_optimizer = Adam(discriminator.parameters(), lr=learning_rate, betas=beta_value)

# Loss setup
BCE_loss = nn.BCELoss().to(device)
Custom_loss = CustomLoss(line_weight=Fronts_weight, l1_weight=L1_weight).to(device)

# Training loop
D_avg_losses = []
G_avg_losses = []

#write config for future use
f = open(f"{target_folder}/config.txt", "w")
f.write(f"epoch: {epoch}\nL1_weight: {L1_weight}\nFronts_weight:{Fronts_weight}\nlearning_rate: {learning_rate}\nbeta_value: {beta_value}\nnum_filters:{num_filters}\nimage_size:{image_size}")
f.close()

print("Training...")
for e in range(epoch):
    print(f"Epoch: {e}")
    D_losses = []
    G_losses = []

    for (src_images, dst_images) in tqdm(train_loader):
        src_images = src_images.to(device)
        dst_images = dst_images.to(device) 

        # Train discriminator with real data
        D_optimizer.zero_grad()
        D_real_decision_unsq = discriminator(src_images, dst_images)
        D_real_decision = D_real_decision_unsq.squeeze()
        D_real_loss = BCE_loss(D_real_decision, torch.ones_like(D_real_decision))
        
        # Train discriminator with fake data
        gen_images = generator(src_images)
        D_fake_decision = discriminator(src_images, gen_images.detach()).squeeze()
        D_fake_loss = BCE_loss(D_fake_decision, torch.zeros_like(D_fake_decision))
        
        D_loss = (D_real_loss + D_fake_loss)
        D_loss.backward()
        D_optimizer.step()
        D_losses.append(D_loss.item())

        # Train generator
        G_optimizer.zero_grad()
        D_fake_decision = discriminator(src_images, gen_images).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, torch.ones_like(D_fake_decision))

        CL_reg = Custom_loss(gen_images, dst_images)
        G_loss = G_fake_loss +CL_reg
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.item())

    # Compute average losses
    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)
    
    # Visualize and save images
    show_tensor_images(D_real_decision_unsq, num_images=1, fn=os.path.join(target_folder, f"dis_{e}.png"))
    show_tensor_images(src_images, num_images=1, fn=os.path.join(target_folder, f"src_{e}.png"))
    show_tensor_images(dst_images, num_images=1, fn=os.path.join(target_folder, f"dst_{e}.png"))
    show_tensor_images(gen_images.cpu().data, num_images=1, fn=os.path.join(target_folder, f"gen_{e}.png"))
    mk_loss_chart(D_avg_losses, G_avg_losses, fn=os.path.join(target_folder, "loss_chart.png"))
    
    # Save models
    torch.save(generator.state_dict(), os.path.join(target_folder, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(target_folder, "discriminator.pth"))