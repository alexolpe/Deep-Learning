import torch
import os
import random
import numpy
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as tvt
import time
from PIL import Image

## Task 3.2---------------------------------------------------------------------------
front = Image.open("front.jpg") 
print("front image: ", front.format, front.size, front.mode)

oblique = Image.open("oblique.jpg") 
print("oblique image: ", oblique.format, oblique.size, oblique.mode)

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


# Compute the similarity of the histograms of 2 images and plot the histograms
def compute_similarity(A, B):
    num_bins = 10
    
    A_to_tensor = tvt.ToTensor()(A)
    B_to_tensor = tvt.ToTensor()(B)
    
    color_channels_A = [A_to_tensor[ch] for ch in range(3)]
    color_channels_B = [B_to_tensor[ch] for ch in range(3)]
    histsA = [torch.histc(color_channels_A[ch],bins=num_bins) for ch in range(3)]
    histsA = [histsA[ch].div(histsA[ch].sum()) for ch in range(3)]
    histsB = [torch.histc(color_channels_B[ch],bins=num_bins) for ch in range(3)]
    histsB = [histsB[ch].div(histsB[ch].sum()) for ch in range(3)]    

    dist_r = wasserstein_distance( histsA[0].cpu().numpy(), histsB[0].cpu().numpy() )
    dist_g = wasserstein_distance( histsA[1].cpu().numpy(), histsB[1].cpu().numpy() )
    dist_b = wasserstein_distance( histsA[2].cpu().numpy(), histsB[2].cpu().numpy() )
    
    dist = (dist_r + dist_g + dist_b)/3 

    # Plotting histsA and histsB with titles
    for ch in range(3):
        plt.subplot(2, 3, ch + 1)
        plt.bar(range(num_bins), histsA[ch].cpu().numpy(), alpha=0.5, color='r', label='histsA')
        plt.bar(range(num_bins), histsB[ch].cpu().numpy(), alpha=0.5, color='b', label='histsB')
        plt.title(f'Histogram - Channel {ch + 1}')
        plt.legend()

    plt.show()
    
    return dist

width, height = front.size
startpoints = [[0,0], [width, 0], [width, height], [0, height]]
#Set the points of the transformation
endpoints = [[0-200,0], [width-200, 0+200], [width, height], [0, height+520]]
perspective_img = tvt.functional.perspective(oblique, startpoints, endpoints, interpolation=3)

perspective_img.save(f"new.jpg")

dist  = compute_similarity(front, oblique)
print("Distance between the 2 original images: ", dist)
dist  = compute_similarity(front, perspective_img)
print("Distance between the front image and the transformed oblique image", dist)

## Task 3.3---------------------------------------------------------------------------
class MyDataset ( torch.utils.data.Dataset ):
    
    def __init__ ( self , root ):
        super().__init__()
        # Obtain meta information (e.g. list of file names )
        self.root = root
        self.filenames = os.listdir(self.root)
        
        # Initialize data augmentation transforms , etc.
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0], [1]),
            tvt.RandomRotation(degrees = 45),
            tvt.RandomHorizontalFlip( p = 0.5),
            tvt.RandomVerticalFlip( p = 0.5),
            tvt.ColorJitter(brightness=.4, hue=.2)
        ])

    def __len__ ( self ):
        # Return the total number of images
        # the number is a place holder only
        return len(self.filenames)

    def __getitem__ ( self , index ):
        # Read an image at index and perform augmentations
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        
        #Apply augmentation
        transformed_img = self.transform(img)
        
        # Return the tuple : ( augmented tensor , integer label )
        return transformed_img, random.randint(0, 10)
    
my_dataset = MyDataset ("bottle_images")
print(len( my_dataset ) )
index = 6
print( my_dataset[ index ][0].shape, my_dataset[ index ][1])

index = 8
print( my_dataset[ index ][0].shape, my_dataset[ index ][1])

##Task 3.4-------------------------------------------------------------------------------
# Create the Dataloader and save the four images from the first batch
my_dataset = MyDataset ("bottle_images")

batch_size = 4
dataloader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )

# Plot the first batch of images
for batch in dataloader :
    images , labels = batch
    for i, image in enumerate(images):
        new = tvt.ToPILImage()(image)
        new.save(f"new/{i}.jpg")
        

# Computing the time needed to process 1000 images by just using Dataset
start_time = time.time()
for _ in range(1000):
    index = random.randint(0, 9)
    img_tensor = my_dataset[ index ][0]
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"By just using Dataset: {elapsed_time} seconds")

# Computing the time needed to process 1000 images by using Dataloader
batch_size = 4
num_workers = 4

my_1000_dataset = MyDataset ("1000_bottle_images")

dataloader = DataLoader(my_1000_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers)
start_time_2 = time.time()
it = 0
for batch in dataloader :
    images , labels = batch
    it += 1
    if it >= (1000/batch_size):
        end_time_2 = time.time()

elapsed_time_2 = end_time_2 - start_time_2
print(f"By using Dataloader: {elapsed_time_2} seconds")

##Task 3.5------------------------------------------------------------------------------------
seed = 60146
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ["PYTHONHASHSEED"] = str(seed)

    
batch_size = 2
dataloader = DataLoader(my_dataset, batch_size = batch_size, shuffle = True )

# Plot the first batch of images
for batch in dataloader :
    images , labels = batch
    for i, image in enumerate(images):
        new = tvt.ToPILImage()(image)
        new.save(f"new/{i}.jpg")
    break