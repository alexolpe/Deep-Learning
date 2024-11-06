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
# THE REST OF THE CODE IS TAKEN FROM THE DLSTUDIO LIBRARY

class Prova1(DLStudio):
    class Prova2(DLStudio.SemanticSegmentation):
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
                    
                    #WE IMPLEMENT THE COMBINED LOSS. WE CREATE A LOSS VECTOR AND SET required_grad=True TO ENSURE BACKPROPAGATION
                    loss = torch.tensor(0.0, requires_grad=True).float().to(self.dl_studio.device)                                  
                    mse_loss = criterion1(output, mask_tensor)  
                    dice_loss = self.dice_loss(preds=output, ground_truth=mask_tensor)
                    loss = mse_loss + 40 * dice_loss
                    loss.backward()
                    
                    optimizer.step()
                    
                    running_loss += loss.item()   
                    running_mse_loss += mse_loss.item()
                    running_dice_loss += dice_loss.item() 
                    
                    if i%500==499:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        
                        avg_loss = running_loss / float(500)
                        avg_mse_loss = running_mse_loss / float(500)
                        avg_dice_loss = running_dice_loss / float(500)
                        
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
            with open('/home/aolivepe/ECE60146/HW7/DLStudio-2.3.6/Examples/dictionary_Combined_scaleDice_40.pkl', 'wb') as archivo:
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
                    if i % 50 == 0:
                        aa= i+1
                        print("\n\n\n\nShowing output for test batch %d: " % (aa))
                        outputs = net(im_tensor)                        
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
                                        elif mask_layer_idx == 3:
                                            if 165 < outputs[batch_im_idx,mask_layer_idx,i,j] < 230:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 4:
                                            if outputs[batch_im_idx,mask_layer_idx,i,j] > 210:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50

                                display_tensor[2*batch_size+batch_size*mask_layer_idx+batch_im_idx,:,:,:]= outputs[batch_im_idx,mask_layer_idx,:,:]
                                
                        #WE MODIFY THE CODE IN ORDER TO SAVE THE TESTING IMAGES WITH THEIR MASKS
                        # self.dl_studio.display_tensor_as_image(
                        #    torchvision.utils.make_grid(display_tensor, nrow=batch_size, normalize=True, padding=2, pad_value=10))
                        image = TF.to_pil_image(torchvision.utils.make_grid(display_tensor, nrow=batch_size, normalize=True, padding=2, pad_value=10))
                        image.save(f"/home/aolivepe/ECE60146/HW7/DLStudio-2.3.6/Examples/testing_imgs_comb_40/{aa}.png")


# WE USE THE NAME OF THE SUPERCLASSES THAT WE HAVE CREATED
dls = Prova1(
                  dataroot = "./../../data/",
                  image_size = [64,64],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 6,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  use_gpu = True,
              )

segmenter = Prova1.Prova2( 
                  dl_studio = dls, 
                  max_num_objects = 5,
              )

dataserver_train = Prova1.Prova2.PurdueShapes5MultiObjectDataset(
                          train_or_test = 'train',
                          dl_studio = dls,
                          segmenter = segmenter,
                          dataset_file = "PurdueShapes5MultiObject-10000-train.gz", 
                        )
dataserver_test = Prova1.Prova2.PurdueShapes5MultiObjectDataset(
                          train_or_test = 'test',
                          dl_studio = dls,
                          segmenter = segmenter,
                          dataset_file = "PurdueShapes5MultiObject-1000-test.gz"
                        )
segmenter.dataserver_train = dataserver_train
segmenter.dataserver_test = dataserver_test

segmenter.load_PurdueShapes5MultiObject_dataset(dataserver_train, dataserver_test)

model = segmenter.mUnet(skip_connections=True, depth=16)
#model = segmenter.mUnet(skip_connections=False, depth=4)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d\n" % number_of_learnable_params)

num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model: %d\n\n" % num_layers)


segmenter.run_code_for_training_for_semantic_segmentation(model)

# import pymsgbox
# response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
# if response == "OK": 
segmenter.run_code_for_testing_semantic_segmentation(model)

