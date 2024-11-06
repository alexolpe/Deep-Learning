import torchvision.transforms as tvt
import cv2
from PIL import Image
import skimage.io as io
from pycocotools.coco import COCO
import os
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import pickle
from skimage.transform import resize

#Data to use for either the training or validation dataset
full_COCO_training_path = "../../../../Downloads/train2017/train2017" 
full_COCO_validation_path = "../../../../Downloads/val2017/val2017" 

our_COCO_training_path = "./HW7_TRAINING_DATASET"
our_COCO_validation_path = "./HW7_VALIDATION_DATASET"

annotations_training_path = "./../HW6/annotations/instances_train2017.json"
annotations_validation_path = "./../HW6/annotations/instances_val2017.json"

coco=COCO(annotations_validation_path)

# Mapping from COCO label to Class indices
classes = ["dog", "cake", "motorcycle"]
catIds = coco.getCatIds(catNms=classes)
categories = coco.loadCats(catIds)
categories.sort(key=lambda x: x['id'])
coco_labels_inverse = {}
for idx, in_class in enumerate(classes):
    for c in categories:
        if c['name'] == in_class:
            coco_labels_inverse[c['id']] = idx

#4: motorcycle, 18: dog, 61: cake
# Save in an array the image ids of the images containing instance of 2 or more of the targeted classes            
imgIds_4_18 = coco.getImgIds(catIds=[4, 18])
imgIds_4_61 = coco.getImgIds(catIds=[4, 61])
imgIds_61_18 = coco.getImgIds(catIds=[61, 18])
imgIds_4_61_18 = coco.getImgIds(catIds=[4, 61, 18])

total_ids = imgIds_4_18 + imgIds_4_61 + imgIds_61_18 + imgIds_4_61_18
print(len(total_ids))

dataset = {}
a = 0
# Iterate 3 times each over the images containing instances of ["dog", "cake", "motorcycle"] and if accomplish the requirements, add it to the dataset
for i, cat_id in enumerate(catIds):
    imgIds = coco.getImgIds(catIds=cat_id)
    
    # Remove the images containing instance of 2 or more of the targeted classes 
    filtered_array = [x for x in imgIds if x not in total_ids]

    for j, img_id in enumerate(filtered_array):
        annIds = coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=False)
        anns = coco.loadAnns(annIds)
        
        # If there is more than one instance of one class in the image we skip it
        if len(anns) != 1:
            continue
        
        # Get the r, g, b arrays to save in the final dictionary resized to 256x256
        img = coco.loadImgs(img_id)[0]
        I = io.imread(os.path.join(full_COCO_validation_path, img['file_name']))
        if len(I.shape) == 2:
            I = skimage.color.gray2rgb(I)
        img_h, img_w = I.shape[0], I.shape[1]
        I = resize(I, (256, 256), anti_aliasing=True, preserve_range=True)
        image = np.uint8(I)
        
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]

        # Reshaping each channel into a 1D array
        r = r.reshape(-1, 1)  # Reshape to (256*256, 1)
        g = g.reshape(-1, 1)
        b = b.reshape(-1, 1)

        area = True
        for i, ann in enumerate(anns):
            # If the area of the bounding box is lower than 200*200 we skip it
            if ann['area']<200*200:
                area=False
                continue
            
            # Get the bounding box and mask resized to 256x256
            [x, y, w, h] = ann['bbox']
            bbox = [x*(256/img_w), y*(256/img_h), w*(256/img_w), h*(256/img_h)]
            mask = coco.annToMask(ann)
            mask = resize(mask, (256, 256), anti_aliasing=True, preserve_range=True)
            
            threshold = 0.5

            # Binarize the array
            mask = (mask > threshold).astype(int)
            
            # Save the mask and the bounding box following the characteristics mentioned at the beginning of section 4
            final_mask = torch.zeros(3, 256, 256)
            if cat_id == 4:
                final_mask[0, :, :] = torch.from_numpy(mask)*50
                bbox_dict = {
                    0:[[x*(256/img_w), y*(256/img_h), x*(256/img_w) + w*(256/img_w), y*(256/img_h) + h*(256/img_h)]], 
                    1:[], 
                    2:[]
                }
            if cat_id == 18:
                final_mask[1, :, :] = torch.from_numpy(mask)*100
                bbox_dict = {
                    0:[], 
                    1:[[x*(256/img_w), y*(256/img_h), x*(256/img_w) + w*(256/img_w), y*(256/img_h) + h*(256/img_h)]], 
                    2:[]
                }
            if cat_id == 61:
                final_mask[2, :, :] = torch.from_numpy(mask)*150
                bbox_dict = {
                    0:[], 
                    1:[], 
                    2:[[x*(256/img_w), y*(256/img_h), x*(256/img_w) + w*(256/img_w), y*(256/img_h) + h*(256/img_h)]]
                }

        if area == False:
            continue
        
        # We create a dictionary called dataset where for each image with index "a" we save the following information
        dataset[a] = {
            0: r, 
            1: g, 
            2: b, 
            3: final_mask, 
            4: bbox_dict
        }
        
        a = a+1
    print(f"**Acumulated** Num of images labeled with cat_id {cat_id}: {a}")

#Save the dictionary in a "pkl" file
with open('validation_dataset.pkl', 'wb') as file:
    pickle.dump(dataset, file)