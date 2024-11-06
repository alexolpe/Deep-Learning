import sys,os,os.path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torchvision.transforms as tvt
from PIL import Image
import torch.nn.functional as F
import copy

from DLStudio import *

os.environ['CUDA_VISIBLE_DEVICES']='5'

## USED THE MYDATASET CLASS USED IN PREVIOUS HOMEWORKS ---------------------------------------------------
class MyDataset ( torch.utils.data.Dataset ):
    
    def __init__ ( self , root ):
        super().__init__()
        # Obtain meta information (e.g. list of file names )
        self.root = root
        self.filenames = os.listdir(self.root)
        
        # Initialize data transforms , etc.
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__ ( self ):
        # Return the total number of images
        # the number is a place holder only
        return len(self.filenames)

    def __getitem__ ( self , index ):
        # Read an image at index and perform processing
        path = os.path.join(self.root, self.filenames[index])
        cls = self.filenames[index].split('_')[0]
        img = Image.open(path)
        
        # Get the normalized tensor
        transformed_img = self.transform(img)
        
        if cls == "boat":
            label = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        elif cls == "couch":
            label = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
        elif cls == "dog":
            label = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        elif cls == "cake":
            label = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        elif cls == "motorcycle":
            label = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Return the tuple : ( normalized tensor, label )
        return transformed_img, label

## COPPIED THE LAST SKIPBLOCK CLASS IMPLEMENTATION PROVIDED BY PROFESSOR KAK---------------------------------------------------
class SkipBlock(nn.Module):
    """
    Class Path:   DLStudio  ->  SkipConnections  ->  SkipBlock
    """            
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlock, self).__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo1 = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.in2out  =  nn.Conv2d(in_ch, out_ch, 1)
        
        if downsample:
            self.downsampler1 = nn.Conv2d(in_ch, in_ch, 1, stride=2)
            self.downsampler2 = nn.Conv2d(out_ch, out_ch, 1, stride=2)

    def forward(self, x):
        identity = x
                                             
        out = self.convo1(x)                              
        out = self.bn1(out)                              
        out = nn.functional.relu(out)
        
        out = self.convo2(out)                              
        out = self.bn2(out)                              
        out = nn.functional.relu(out)
        
        if self.downsample:
            identity = self.downsampler1(identity)
            out = self.downsampler2(out)
            
        if self.skip_connections:
            if (self.in_ch == self.out_ch) and (self.downsample is False):
                out = out + identity
            elif (self.in_ch != self.out_ch) and (self.downsample is False):
                identity = self.in2out( identity )                             ###  <<<<  from  Cheng-Hao Chen
                out = out + identity
            elif (self.in_ch != self.out_ch) and (self.downsample is True):
                out = out + torch.cat((identity, identity), dim=1)
                
        return out

## COPPIED THE FIRST BMENET CLASS IMPLEMENTATION AVAILABLE IN DLSTUDIO AND EXTEND IT TO CREATE HW5NET ---------------------------------------------------
class HW5Net(nn.Module):
    """
    Class Path:   DLStudio  ->  SkipConnections  ->  HW5Net
    """
    def __init__(self, skip_connections=True, depth=32):
        super(HW5Net, self).__init__()
        if depth not in [8, 16, 32, 64]:
            sys.exit("HW5Net has been tested for depth for only 8, 16, 32, and 64")
        self.depth = depth // 8
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.skip64_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip64_arr.append(SkipBlock(64, 64,skip_connections=skip_connections))
        self.skip64ds = SkipBlock(64, 64,downsample=True, skip_connections=skip_connections)
        self.skip64to128 = SkipBlock(64, 128,skip_connections=skip_connections )
        
        self.skip128_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip128_arr.append(SkipBlock(128, 128,skip_connections=skip_connections))
        self.skip128ds = SkipBlock(128, 128,downsample=True, skip_connections=skip_connections)
        self.skip128to256 = SkipBlock(128, 256,skip_connections=skip_connections )
        
        self.skip256_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip256_arr.append(SkipBlock(256, 256, skip_connections=skip_connections))
        self.skip256ds = SkipBlock(256,256,downsample=True, skip_connections=skip_connections)

        self.fc1 =  nn.Linear(1024, 500)
        self.fc2 =  nn.Linear(500, 5)
        
    # I implement the forward method as an extension of the given BMEnet given. I downsample the input images one more time and I also increase the number of channels one more time. 
    # Finally I adjust the arguments of the linear layers
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x))) 
                 
        for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            x = skip64(x)                
        x = self.skip64ds(x)
        for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
            x = skip64(x)                
        x = self.skip64ds(x)       
        x = self.skip64to128(x)
        
        for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
            x = skip128(x)                
        x = self.skip128ds(x)
        for i,skip128 in enumerate(self.skip128_arr[self.depth//4:]):
            x = skip128(x)                
        x = self.skip128ds(x)       
        x = self.skip128to256(x)
        
        for i,skip256 in enumerate(self.skip256_arr[:self.depth//4]):
            x = skip256(x)                
        for i,skip256 in enumerate(self.skip256_arr[self.depth//4:]):
            x = skip256(x)
                            
        x  =  x.view( x.shape[0], - 1 )
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x  
    
## SCRIPT TO RUN THE TRAINING ---------------------------------------------------
device = "cuda:0"

# Dataloader
my_dataset = MyDataset ("../../../COCOTraining")
batch_size = 4
train_data_loader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )


# List where the loss values will be stored
loss_net3 = []

net3 = HW5Net()
# Training routine provided by the assignment
net3 = net3.to(device)

# Print number of learnable layers
num_layers = len( list ( net3.parameters () ) )
print("number of learnable layers: ", num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net3.parameters(), lr=5e-4, betas=(0.9, 0.99))
epochs = 60
for epoch in tqdm(range(epochs)):       
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net3(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            # print("[epoch: %d, batch: %5d] loss: %.3f" \
            # % (epoch + 1, i + 1, running_loss / 100))
            loss_net3.append(running_loss/100)
            running_loss = 0.0

# Save the learned parameters of the model to do inference afterwards
torch.save(net3.state_dict(), './net5_replicate_60epochs.pth')

## PLOT TRAINING LOSS----------------------------------------------------------------------------------
plt.plot(loss_net3)

plt.legend(["loss_net3"])

# Adding labels and title
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss comparisson for the 3 networks')

# Display the plot
plt.show()

## TESTING AND CONFUSSION MATRIX (We run this code for each network)---------------------------------------------------
device = "cuda:0"

# Dataloader loading the Validation dataset
my_dataset = MyDataset("../COCOValidation")
batch_size = 4
train_data_loader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )

# Lists where the labels will be stored for each of the images from the Validation dataset
predictions = []
real_labels = []

# Load the trained weights
net3 = HW5Net()
net3.load_state_dict(torch.load('net5_replicate_60epochs.pth'))
net3 = net3.to(device)

# Set the model to evaluation mode
net3.eval()

# Get the predicted label and the real label from each image and store them to the lists mentioned before
for i, data in enumerate(train_data_loader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = net3(inputs)
    outputs = outputs.split(1)
    labels = labels.split(1)
    for lbl in labels:
        real_labels.append(torch.argmax(lbl.squeeze()).item())
    for pred in outputs:
        predictions.append(torch.argmax(pred.squeeze()).item())

# Compute the confusion matrix
cm = confusion_matrix(real_labels, predictions)

# Create a heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=['boat','couch','dog', 'cake', 'motorcycle'], yticklabels=['boat','couch','dog', 'cake', 'motorcycle'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix Net3')
plt.show()

# Print classification report for additional metrics like accuracy
print("Classification Report:\n", classification_report(real_labels, predictions))
