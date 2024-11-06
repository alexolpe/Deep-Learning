import torchvision.transforms as tvt
from PIL import Image
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

## DEFINITION OF THE DATASET ------------------------------------------------------------------------
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
    
## CNN TASK 1------------------------------------------------------------------------------------------
class HW4Net1(nn.Module):
    def __init__(self):
        super(HW4Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*14*14, 64)
        self.fc2 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = "cuda:0"

# Dataloader
my_dataset = MyDataset ("../COCOTraining")
batch_size = 4
train_data_loader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )

net1 = HW4Net1()

# Print num of learnable parameters
total_params = sum(p.numel() for p in net1.parameters())
print(f"Number of learnable parameters: {total_params}")

# List where the loss values will be stored
loss_net1 = []

# Training routine provided by the assignment
net1 = net1.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
net1.parameters(), lr=1e-3, betas=(0.9, 0.99))
epochs = 15
for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            # print("[epoch: %d, batch: %5d] loss: %.3f" \
            # % (epoch + 1, i + 1, running_loss / 100))
            loss_net1.append(running_loss/100)
            running_loss = 0.0

# Save the learned parameters of the model to do inference afterwards            
torch.save(net1.state_dict(), './net1_normalization.pth')

## CNN TASK 2------------------------------------------------------------------------------------------
class HW4Net2(nn.Module):
    def __init__(self):
        super(HW4Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*16*16, 64) #Now here there will be an input of 16x16x32 features
        self.fc2 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = "cuda:0"

# Dataloader
my_dataset = MyDataset ("../COCOTraining")
batch_size = 4
train_data_loader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )

net2 = HW4Net2()

# Print num of learnable parameters
total_params = sum(p.numel() for p in net2.parameters())
print(f"Number of learnable parameters: {total_params}")

# List where the loss vañues will be stored
loss_net2 = []

# Training routine provided by the assignment
net2 = net2.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
net2.parameters(), lr=1e-3, betas=(0.9, 0.99))
epochs = 15
for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            # print("[epoch: %d, batch: %5d] loss: %.3f" \
            # % (epoch + 1, i + 1, running_loss / 100))
            loss_net2.append(running_loss/100)
            running_loss = 0.0

# Save the learned parameters of the model to do inference afterwards
torch.save(net2.state_dict(), './net2_normalization.pth')

## CNN TASK 3------------------------------------------------------------------------------------------
class HW4Net3(nn.Module):
    def __init__(self):
        super(HW4Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv_repeat = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*16*16, 64) #Now here there will be an input of 16x16x32 features
        self.fc2 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = F.relu(self.conv_repeat(x))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = "cuda:0"

# Dataloader
my_dataset = MyDataset ("../COCOTraining")
batch_size = 4
train_data_loader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )

net3 = HW4Net3()

# Print num of learnable parameters
total_params = sum(p.numel() for p in net3.parameters())
print(f"Number of learnable parameters: {total_params}")

# List where the loss values will be stored
loss_net3 = []

# Training routine provided by the assignment
net3 = net3.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
net3.parameters(), lr=1e-3, betas=(0.9, 0.99))
epochs = 15
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
torch.save(net3.state_dict(), './net3_normalization.pth')

## PLOT TRAINING LOSS----------------------------------------------------------------------------------
plt.plot(loss_net1)
plt.plot(loss_net2)
plt.plot(loss_net3)

plt.legend(["loss_net1", "loss_net2", "loss_net3"])

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
net3 = HW4Net3()
net3.load_state_dict(torch.load('net3_normalization.pth'))
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