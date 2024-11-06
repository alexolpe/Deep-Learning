import random
import numpy
import torch
import os
import sys
import torch
import torch.nn as nn
import torchvision                  
import torchvision.transforms as tvt
import glob  
import torch.optim as optim
import time 
import imageio                                                                                                        
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvtF
import numpy as np

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

# Discriminator network definition
class Discriminator(nn.Module):
    def __init__(self, num_colors=3, depths=128, image_size=64):
        super(Discriminator, self).__init__()

        # Convolutional layers with LeakyReLU activations
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_colors, depths, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(depths, depths * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(depths * 2, depths * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(depths * 4, depths * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(depths * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.final_size = image_size // 16

    def forward(self, x):
        # Forward pass through convolutional layers
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        return conv5_out.squeeze()

# Generator network definition
class Generator(nn.Module):
    def __init__(self, num_noises=100, num_colors=3, depths=128, image_size=64):
        super(Generator, self).__init__()

        if image_size % 16 != 0:
            raise Exception("Size of the image must be divisible by 16")
        self.final_size = image_size // 16
        self.depths = depths

        # Linear layer and convolutional layers with ReLU activations
        self.lin = nn.Linear(num_noises, depths * 8 * self.final_size * self.final_size)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(depths * 8, depths * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(depths * 4, depths * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(depths * 2, depths, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(depths * 2 + depths, depths, 4, 2, 1, bias=False),  # Skip connection
            nn.BatchNorm2d(depths),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(depths, num_colors, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Forward pass through linear and convolutional layers
        lin_out = self.lin(x)
        lin_out = lin_out.view(-1, self.depths * 8, self.final_size, self.final_size)
        conv1_out = self.conv1(lin_out)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        resized_conv2_out = nn.functional.interpolate(conv2_out, size=(conv3_out.size(2), conv3_out.size(3)), mode='bilinear', align_corners=False)
        cat_input = torch.cat((conv3_out, resized_conv2_out), dim=1)  # Concatenate with conv2_out
        conv4_out = self.conv4(cat_input)
        conv5_out = self.conv5(conv4_out)
        conv5_out = nn.functional.interpolate(conv5_out, size=(64, 64), mode='bilinear', align_corners=False)
        return conv5_out
    
## Defining variables ------------------------------------------------------
dataroot = "./../../data/celeba_dataset_64x64/"
image_size = [64,64]
batch_size = 32
num_workers = 2
dir_name_for_results = "results_mix"
device = "cuda:1"

learning_rate = 0.0002
beta1 = 0.5
epochs = 30
## -------------------------------------------------------------------------

## Dataset and dataloader --------------------------------------------------
dataset = torchvision.datasets.ImageFolder(root=dataroot,       
                                        transform = tvt.Compose([                 
                                                    tvt.Resize(image_size),             
                                                    tvt.CenterCrop(image_size),         
                                                    tvt.ToTensor(),                     
                                                    tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),         
                                        ]))
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
## --------------------------------------------------------------------------               
                
discriminator =  Discriminator()
generator =  Generator()

nz = 100
netD = discriminator.to(device)
netG = generator.to(device)
netG.load_state_dict(torch.load('./netG.pth'))

# Generate 2048 samples
for i in range(2048):
    fixed_noise = torch.randn(1, nz, device=device)          
    fake = netG(fixed_noise).detach().cpu()
    img = tvtF.to_pil_image(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True)) 
    imageio.imwrite(f"results_mygan_Gskip_Dskip_experiments_final/image_{i}.png", img)