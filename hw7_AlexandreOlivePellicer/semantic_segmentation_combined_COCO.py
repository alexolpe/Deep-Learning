#!/usr/bin/env python

##  semantic_segmentation.py

"""
This script should be your starting point if you wish to learn how to use the
mUnet neural network for semantic segmentation of images.  As mentioned elsewhere in
the main documentation page, mUnet assigns an output channel to each different type of
object that you wish to segment out from an image. So, given a test image at the
input to the network, all you have to do is to examine each channel at the output for
segmenting out the objects that correspond to that output channel.
"""

import random
import numpy
import torch
import os, sys

import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox
import time
import logging
import torchvision.transforms.functional as TF


from DLStudio import *

# WE CREATE SUPERCLASSES TO OVERWRITE THE FOLLOWING FUNCTIONS FROM THE DLSTUDIO LIBRARY:
    
# run_code_for_training_for_semantic_segmentation(self, net)
#     WE IMPLEMENT THE COMBINED LOSS AND SAVE THE RUNNING LOSS VALUES TO DISPLAY THEM LATER IN PLOTS
    
# run_code_for_testing_semantic_segmentation(self, net)
#     WE MODIFY THE CODE IN ORDER TO SAVE THE TESTING IMAGES WITH THEIR MASKS
    
# WE ADD THE METHOD dice_loss IMPLEMENTED FOLLOWING THE INSTRUCTIONS FROM SECTION 3.3

# WE CREATE A SUPERCLASS OF THE DATASET CLASS TO OVERWRITE ITS METHODS

# WE CREATE A SUPERCLASS OF THE mUnet CLASS TO OVERWRITE ITS METHODS

# THE REST OF THE CODE IS TAKEN FROM THE DLSTUDIO LIBRARY

class Prova1(DLStudio):
    class Prova2(DLStudio.SemanticSegmentation):
        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, dl_studio, segmenter, train_or_test, dataset_file):
                super(Prova1.Prova2.MyDataset, self).__init__()
                max_num_objects = segmenter.max_num_objects
                if train_or_test == 'train':
                    print("\nLoading training data from torch saved file")
                    
                    # Load dictionary with training dataset
                    with open('/home/aolivepe/ECE60146/HW7/training_dataset.pkl', 'rb') as file:
                        self.dataset = pickle.load(file)
                    
                    self.label_map = {
                        'motorcycle': 50,
                        'dog': 100, 
                        'cake': 150
                    }
                                        
                    self.num_shapes = len(self.label_map)
                    self.image_size = dl_studio.image_size
                else:
                    # Load dictionary with testing dataset
                    with open('/home/aolivepe/ECE60146/HW7/validation_dataset.pkl', 'rb') as file:
                        self.dataset = pickle.load(file)
                    
                    self.label_map = {
                        'motorcycle': 50,
                        'dog': 100, 
                        'cake': 150
                    }
                    
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = {
                        50: 'motorcycle',
                        100: 'dog', 
                        150: 'cake'
                    }
                    self.num_shapes = len(self.class_labels)
                    self.image_size = dl_studio.image_size

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                
                # Creating image
                image_size = self.image_size
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(image_size[0],image_size[1]), g.reshape(image_size[0],image_size[1]), b.reshape(image_size[0],image_size[1])
                im_tensor = torch.zeros(3,image_size[0],image_size[1], dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                
                # Getting mask
                mask_array = np.array(self.dataset[idx][3])
                max_num_objects = len( mask_array[0] )
                mask_tensor = torch.from_numpy(mask_array)
                
                
                mask_val_to_bbox_map =  self.dataset[idx][4]
                max_bboxes_per_entry_in_map = max([ len(mask_val_to_bbox_map[key]) for key in mask_val_to_bbox_map ])
                ##  The first arg 5 is for the number of bboxes we are going to need. If all the
                ##  shapes are exactly the same, you are going to need five different bbox'es.
                ##  The second arg is the index reserved for each shape in a single bbox
                bbox_tensor = torch.zeros(max_num_objects,self.num_shapes,4, dtype=torch.float)
                for bbox_idx in range(max_bboxes_per_entry_in_map):
                    for key in mask_val_to_bbox_map:
                        if len(mask_val_to_bbox_map[key]) == 1:
                            if bbox_idx == 0:
                                bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                        elif len(mask_val_to_bbox_map[key]) > 1 and bbox_idx < len(mask_val_to_bbox_map[key]):
                            bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                sample = {'image'        : im_tensor, 
                          'mask_tensor'  : mask_tensor,
                          'bbox_tensor'  : bbox_tensor }
                return sample
                
        class MymUnet(nn.Module):
            # EXTENSION OF THE mUnet PROVIDED IN DLSTUDIO BUILT FOLLOWING THE EXPLANATIONS FROM SECTION 4.1.
            def __init__(self, skip_connections=True, depth=16):
                super(Prova1.Prova2.MymUnet, self).__init__()
                self.depth = depth // 2
                self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
                
                ##  For the DN arm of the U:
                self.bn1DN  = nn.BatchNorm2d(64)
                self.bn2DN  = nn.BatchNorm2d(128)
                self.bn3DN  = nn.BatchNorm2d(256)
                self.bn4DN  = nn.BatchNorm2d(512)
                
                self.skip64DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(64, 64, skip_connections=skip_connections))
                self.skip64dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 64,   downsample=True, skip_connections=skip_connections)
                
                self.skip64to128DN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 128, skip_connections=skip_connections )
                
                self.skip128DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(128, 128, skip_connections=skip_connections))
                self.skip128dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(128,128, downsample=True, skip_connections=skip_connections)
                
                self.skip128to256DN = DLStudio.SemanticSegmentation.SkipBlockDN(128, 256, skip_connections=skip_connections )
                
                self.skip256DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip256DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(256, 256, skip_connections=skip_connections))
                self.skip256dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(256,256, downsample=True, skip_connections=skip_connections)
                
                self.skip256to512DN = DLStudio.SemanticSegmentation.SkipBlockDN(256, 512, skip_connections=skip_connections )
                
                self.skip512DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip512DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(512, 512, skip_connections=skip_connections))
                self.skip512dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(512,512, downsample=True, skip_connections=skip_connections)
                
                
                ##  For the UP arm of the U:
                self.bn1UP  = nn.BatchNorm2d(512)
                self.bn2UP  = nn.BatchNorm2d(256)
                self.bn3UP  = nn.BatchNorm2d(128)
                self.bn4UP  = nn.BatchNorm2d(64)
                
                self.skip64UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, skip_connections=skip_connections))
                self.skip64usUP = DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, upsample=True, skip_connections=skip_connections)
                
                self.skip128to64UP = DLStudio.SemanticSegmentation.SkipBlockUP(128, 64, skip_connections=skip_connections )
                
                self.skip128UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(128, 128, skip_connections=skip_connections))
                self.skip128usUP = DLStudio.SemanticSegmentation.SkipBlockUP(128,128, upsample=True, skip_connections=skip_connections)
                
                self.skip256to128UP = DLStudio.SemanticSegmentation.SkipBlockUP(256, 128, skip_connections=skip_connections )
                
                self.skip256UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip256UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(256, 256, skip_connections=skip_connections))
                self.skip256usUP = DLStudio.SemanticSegmentation.SkipBlockUP(256,256, upsample=True, skip_connections=skip_connections)
                
                self.skip512to256UP = DLStudio.SemanticSegmentation.SkipBlockUP(512, 256, skip_connections=skip_connections )
                
                self.skip512UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip512UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(512, 512, skip_connections=skip_connections))
                self.skip512usUP = DLStudio.SemanticSegmentation.SkipBlockUP(512,512, upsample=True, skip_connections=skip_connections)
                
                self.conv_out = nn.ConvTranspose2d(64, 3, 3, stride=2,dilation=2,output_padding=1,padding=2)

            def forward(self, x):
                ##  Going down to the bottom of the U:
                x = nn.MaxPool2d(2,2)(nn.functional.relu(self.conv_in(x))) 
                         
                for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
                    x = skip64(x)                
                num_channels_to_save1 = x.shape[1] // 2
                save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
                x = self.skip64dsDN(x)
                for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1DN(x)
                num_channels_to_save2 = x.shape[1] // 2
                save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
                
                x = self.skip64to128DN(x)
                
                for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
                    x = skip128(x) 
                num_channels_to_save3 = x.shape[1] // 2
                save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
                x = self.skip128dsDN(x)
                for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
                    x = skip128(x)                
                x = self.bn2DN(x)
                num_channels_to_save4 = x.shape[1] // 2
                save_for_upside_4 = x[:,:num_channels_to_save4,:,:].clone()
                
                x = self.skip128to256DN(x)
                
                for i,skip256 in enumerate(self.skip256DN_arr[:self.depth//4]):
                    x = skip256(x) 
                num_channels_to_save5 = x.shape[1] // 2
                save_for_upside_5 = x[:,:num_channels_to_save5,:,:].clone()
                x = self.skip256dsDN(x)
                for i,skip256 in enumerate(self.skip256DN_arr[self.depth//4:]):
                    x = skip256(x)                
                x = self.bn3DN(x)
                num_channels_to_save6 = x.shape[1] // 2
                save_for_upside_6 = x[:,:num_channels_to_save6,:,:].clone()
                
                x = self.skip256to512DN(x)
                
                for i,skip512 in enumerate(self.skip512DN_arr[:self.depth//4]):
                    x = skip512(x)                
                x = self.bn4DN(x)
                num_channels_to_save7 = x.shape[1] // 2
                save_for_upside_7 = x[:,:num_channels_to_save7,:,:].clone()
                for i,skip512 in enumerate(self.skip512DN_arr[self.depth//4:]):
                    x = skip512(x)                
                x = self.skip512dsDN(x)
                
                
                ## Coming up from the bottom of U on the other side:
                x = self.skip512usUP(x) 
                         
                for i,skip512 in enumerate(self.skip512UP_arr[:self.depth//4]):
                    x = skip512(x)    
                x[:,:num_channels_to_save7,:,:] =  save_for_upside_7
                x = self.bn1UP(x)
                for i,skip512 in enumerate(self.skip512UP_arr[:self.depth//4]):
                    x = skip512(x) 
                    
                x = self.skip512to256UP(x)
                
                for i,skip256 in enumerate(self.skip256UP_arr[self.depth//4:]):
                    x = skip256(x)  
                x[:,:num_channels_to_save6,:,:] =  save_for_upside_6
                x = self.bn2UP(x)
                x = self.skip256usUP(x)
                for i,skip256 in enumerate(self.skip256UP_arr[:self.depth//4]):
                    x = skip256(x)     
                x[:,:num_channels_to_save5,:,:] =  save_for_upside_5
                
                x = self.skip256to128UP(x)
                
                for i,skip128 in enumerate(self.skip128UP_arr[self.depth//4:]):
                    x = skip128(x)  
                x[:,:num_channels_to_save4,:,:] =  save_for_upside_4
                x = self.bn3UP(x)
                x = self.skip128usUP(x)
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)     
                x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                
                x = self.skip128to64UP(x)
                
                for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
                    x = skip64(x)  
                x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                x = self.bn4UP(x)
                x = self.skip64usUP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
                    x = skip64(x)     
                x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                
                x = self.conv_out(x)
                return x
            
        # WE ADD THE METHOD dice_loss IMPLEMENTED FOLLOWING THE INSTRUCTIONS FROM SECTION 3.3        
        def dice_loss (self, preds : torch.Tensor, ground_truth : torch.Tensor, epsilon =1e-6 ):
            """
            inputs :
                preds : predicted mask
                ground_truth : ground truth mask
                epsilon ( float ): prevents division by zero
            returns :
                dice_loss
            """
            
            # Step 1: Compute Dice Coefficient.
            numerator = torch.sum(preds * ground_truth, dim=(2, 3))
            denominator = torch.sum(preds * preds, dim=(2, 3)) + torch.sum(ground_truth * ground_truth, dim=(2, 3))
            
            # Step 2: Compute dice_coefficient
            dice_coefficient = (2 * numerator) / (denominator + epsilon)
            
            # Step 3: Compute dice_loss
            dice_loss = 1 - dice_coefficient
        
            return dice_loss.mean()
            
        def run_code_for_training_for_semantic_segmentation(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            optimizer = optim.SGD(net.parameters(), 
                            lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            
            criterion1_loss = []
            criterion2_loss = []
            criterion3_loss = []
            criterion1 = nn.MSELoss()

            
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                running_mse_loss = 0.0
                running_dice_loss = 0.0
                for i, data in enumerate(self.train_dataloader):  
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    
                    # print("output: ", output.shape, torch.max(output), torch.min(output))
                    # print("mask_tensor: ", mask_tensor.shape, torch.max(mask_tensor), torch.min(mask_tensor))
                    
                    #WE IMPLEMENT THE COMBINED LOSS. WE CREATE A LOSS VECTOR AND SET required_grad=True TO ENSURE BACKPROPAGATION
                    loss = torch.tensor(0.0, requires_grad=True).float().to(self.dl_studio.device)                                  
                    mse_loss = criterion1(output, mask_tensor)  
                    dice_loss = self.dice_loss(preds=output, ground_truth=mask_tensor)
                    loss = mse_loss + 80 * dice_loss
                    loss.backward()
                    
                    optimizer.step()
                    
                    running_loss += loss.item()   
                    running_mse_loss += mse_loss.item()
                    running_dice_loss += dice_loss.item() 
                    
                    if i%50==49:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        
                        avg_loss = running_loss / float(50)
                        avg_mse_loss = running_mse_loss / float(50)
                        avg_dice_loss = running_dice_loss / float(50)
                        
                        #WE SAVE THE RUNNING LOSS VALUES TO DISPLAY THEM LATER IN PLOTS
                        criterion1_loss.append(running_loss)
                        criterion2_loss.append(running_mse_loss)
                        criterion3_loss.append(running_dice_loss)
                        
                        print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   loss: %.3f, MSE loss: %.3f, Dice loss: %.3f" % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss, avg_mse_loss, avg_dice_loss))
                        FILE1.write("%.3f\n" % avg_loss)
                        FILE1.flush()
                        
                        running_loss = 0.0
                        running_mse_loss = 0.0
                        running_dice_loss = 0.0
                        
            print("\nFinished Training\n")
            self.save_model(net)
            
            dictionary_losses = {}

            nombre_imagen = 'yes'
            dictionary_losses[nombre_imagen] = {
                'criterion1': criterion1_loss,
                'criterion2': criterion2_loss,
                'criterion3': criterion3_loss,
            }
            
            with open('/home/aolivepe/ECE60146/HW7/DLStudio-2.3.6/Examples/dictionary_Combined_scaleDice_80_COCO.pkl', 'wb') as archivo:
                pickle.dump(dictionary_losses, archivo)


        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


        def run_code_for_testing_semantic_segmentation(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            batch_size = self.dl_studio.batch_size
            image_size = self.dl_studio.image_size
            max_num_objects = self.max_num_objects
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    if i % 3 == 0:
                        aa= i+1
                        print("\n\n\n\nShowing output for test batch %d: " % (aa))
                        outputs = net(im_tensor)
                        print("output testing: ", outputs.shape)                        
                        ## In the statement below: 1st arg for batch items, 2nd for channels, 3rd and 4th for image size
                        output_bw_tensor = torch.zeros(batch_size,1,image_size[0],image_size[1], dtype=float)
                        for image_idx in range(batch_size):
                            for layer_idx in range(max_num_objects): 
                                for m in range(image_size[0]):
                                    for n in range(image_size[1]):
                                        output_bw_tensor[image_idx,0,m,n]  =  torch.max( outputs[image_idx,:,m,n] )
                        display_tensor = torch.zeros(7 * batch_size,3,image_size[0],image_size[1], dtype=float)
                        for idx in range(batch_size):
                            for bbox_idx in range(max_num_objects):   
                                bb_tensor = bbox_tensor[idx,bbox_idx]
                                for k in range(max_num_objects):
                                    i1 = int(bb_tensor[k][1])
                                    i2 = int(bb_tensor[k][3])
                                    j1 = int(bb_tensor[k][0])
                                    j2 = int(bb_tensor[k][2])
                                    
                                    # I CAN PROBABLY REMOVE THIS
                                    if i1 > 255:
                                        i1 = 255
                                    if i2 > 255:
                                        i2 = 255
                                    if j1 > 255:
                                        j1 = 255
                                    if j2 > 255:
                                        j2 = 255
                                        
                                    output_bw_tensor[idx,0,i1:i2,j1] = 255
                                    output_bw_tensor[idx,0,i1:i2,j2] = 255
                                    output_bw_tensor[idx,0,i1,j1:j2] = 255
                                    output_bw_tensor[idx,0,i2,j1:j2] = 255
                                    im_tensor[idx,0,i1:i2,j1] = 255
                                    im_tensor[idx,0,i1:i2,j2] = 255
                                    im_tensor[idx,0,i1,j1:j2] = 255
                                    im_tensor[idx,0,i2,j1:j2] = 255
                        display_tensor[:batch_size,:,:,:] = output_bw_tensor
                        display_tensor[batch_size:2*batch_size,:,:,:] = im_tensor

                        for batch_im_idx in range(batch_size):
                            for mask_layer_idx in range(max_num_objects):
                                for i in range(image_size[0]):
                                    for j in range(image_size[1]):
                                        if mask_layer_idx == 0:
                                            #SINCE WE ARE WORKING ONLY WITH 3 CLASSES, WE REMOVED THE OTHER 2 LEVELS
                                            if 25 < outputs[batch_im_idx,mask_layer_idx,i,j] < 85:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 1:
                                            if 65 < outputs[batch_im_idx,mask_layer_idx,i,j] < 135:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 2:
                                            if 115 < outputs[batch_im_idx,mask_layer_idx,i,j] < 185:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50

                                display_tensor[2*batch_size+batch_size*mask_layer_idx+batch_im_idx,:,:,:]= outputs[batch_im_idx,mask_layer_idx,:,:]
                        #WE MODIFY THE CODE IN ORDER TO SAVE THE TESTING IMAGES WITH THEIR MASKS
                        # self.dl_studio.display_tensor_as_image(
                        #    torchvision.utils.make_grid(display_tensor, nrow=batch_size, normalize=True, padding=2, pad_value=10))
                        image = TF.to_pil_image(torchvision.utils.make_grid(display_tensor, nrow=batch_size, normalize=True, padding=2, pad_value=10))
                        image.save(f"/home/aolivepe/ECE60146/HW7/DLStudio-2.3.6/Examples/testing_imgs_comb_300_COCO/{aa}.png")

# WE USE THE NAME OF THE SUPERCLASSES THAT WE HAVE CREATED
dls = Prova1(
                  dataroot = "./../../data/",
                  image_size = [256,256],
                  path_saved_model = "./saved_model_Dice_COCO_300",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 50,
                  batch_size = 4,
                  classes = ('motorcycle','dog','cake'),
                  use_gpu = True,
              )

segmenter = Prova1.Prova2( 
                  dl_studio = dls, 
                  max_num_objects = 3,
              )

dataserver_train = Prova1.Prova2.MyDataset(
                          train_or_test = 'train',
                          dl_studio = dls,
                          segmenter = segmenter,
                          dataset_file = "PurdueShapes5MultiObject-10000-train.gz", 
                        )
dataserver_test = Prova1.Prova2.MyDataset(
                          train_or_test = 'test',
                          dl_studio = dls,
                          segmenter = segmenter,
                          dataset_file = "PurdueShapes5MultiObject-1000-test.gz"
                        )
segmenter.dataserver_train = dataserver_train
segmenter.dataserver_test = dataserver_test

#Create dataloaders
segmenter.load_PurdueShapes5MultiObject_dataset(dataserver_train, dataserver_test)

model = segmenter.MymUnet(skip_connections=True, depth=16)
#model = segmenter.mUnet(skip_connections=False, depth=4)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d\n" % number_of_learnable_params)

num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model: %d\n\n" % num_layers)


segmenter.run_code_for_training_for_semantic_segmentation(model)
# model.load_state_dict(torch.load("/home/aolivepe/ECE60146/HW7/DLStudio-2.3.6/Examples/saved_model"))

print("Start Testing")
# import pymsgbox
# response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
# if response == "OK": 
segmenter.run_code_for_testing_semantic_segmentation(model)
