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
import imageio                                                                                                        
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvtF
import numpy as np

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
dir_name_for_results = "final_experiment"
device = "cuda:0"

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

# Function for initializing weights of the network
def weights_init(m):        
    """
    Uses the DCGAN initializations for the weights
    """
    classname = m.__class__.__name__     
    if classname.find('Conv') != -1:         
        nn.init.normal_(m.weight.data, 0.0, 0.02)      
    elif classname.find('BatchNorm') != -1:         
        nn.init.normal_(m.weight.data, 1.0, 0.02)       
        nn.init.constant_(m.bias.data, 0) 
                
# Creating instances of discriminator and generator                
discriminator =  Discriminator()
generator =  Generator()

nz = 100
netD = discriminator.to(device)
netG = generator.to(device)

# Applying weight initialization to discriminator and generator
netD.apply(weights_init)
netG.apply(weights_init)

fixed_noise = torch.randn(batch_size, nz, device=device)          

real_label = 1   
fake_label = 0        

optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))    
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

criterion = nn.BCELoss()

G_losses = []                               
D_losses = []                               
           
# Training logic. Inspired from the implementation available in the "AdversarialLearning.py" file from DL-Studio
for epoch in range(epochs):        
    g_losses_per_print_cycle = []     
    d_losses_per_print_cycle = []       
    for i, data in enumerate(train_dataloader, 0):         

        netD.zero_grad()
        
        real_images_in_batch = data[0].to(device) # Get real images
        b_size = real_images_in_batch.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_images_in_batch).view(-1) # Forward pass real images through discriminator
        lossD_for_reals = criterion(output, label) # Calculate loss for real images
        lossD_for_reals.backward()
        
        noise = torch.randn(b_size, nz, device=device)
        fakes = netG(noise) # Generate fake images
        label.fill_(fake_label)
        output = netD(fakes.detach()).view(-1) # Forward pass fake images through discriminator
        lossD_for_fakes = criterion(output, label) # Calculate loss for fake images
        lossD_for_fakes.backward()
        
        lossD = lossD_for_reals + lossD_for_fakes # Total discriminator loss
        d_losses_per_print_cycle.append(lossD) 
        optimizerD.step()

        netG.zero_grad() # Reset gradients of the generator
        label.fill_(real_label)    
        output = netD(fakes).view(-1) # Forward pass fake images through discriminator
        lossG = criterion(output, label) # Calculate generator loss
        g_losses_per_print_cycle.append(lossG)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 99:
            mean_D_loss = torch.mean(torch.FloatTensor(d_losses_per_print_cycle)) # Calculate mean discriminator loss
            mean_G_loss = torch.mean(torch.FloatTensor(g_losses_per_print_cycle)) # Calculate mean generator loss
            print("[epoch=%d/%d   iter=%4d]     mean_D_loss=%7.4f    mean_G_loss=%7.4f" % ((epoch+1),epochs,(i+1),mean_D_loss,mean_G_loss))   
            d_losses_per_print_cycle = []
            g_losses_per_print_cycle = []

        G_losses.append(lossG.item()) 
        D_losses.append(lossD.item()) 

# Save generator model
torch.save(netG.state_dict(), './netG.pth')
                 
# Plot generator and discriminator losses during training
plt.figure(figsize=(10,5))    
plt.title("Generator and Discriminator Loss During Training")    
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()    
plt.savefig(dir_name_for_results + "/gen_and_disc_loss_training.png")
                            


