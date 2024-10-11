from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
#import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler

from scipy import sparse 

OUTPUT_RESOLUTION = 512 
NUM_OF_BLENDWEIGHT_MAPS_TO_USE = 4  
prefix = '<< path to directory >>'





class ThumanDatasetNovel(Dataset):
    def __init__(self, size, split, useBlendweights=False, useSequentialSmplx=False, justFrontal=False ):

        self.img_size = size
        self.useBlendweights = useBlendweights
        self.useSequentialSmplx = useSequentialSmplx
        self.justFrontal = justFrontal

        if split == 'train':
            self.subject_list = np.loadtxt("{0}/getTestSet/train_set_list_trainSubset.txt".format(prefix), dtype=str)
            self.is_train = True 

        elif split == 'val':
            self.subject_list = np.loadtxt("{0}/getTestSet/train_set_list_validationSubset.txt".format(prefix), dtype=str)
            self.is_train = False 

        elif split == 'test':
            self.subject_list = np.loadtxt("{0}/getTestSet/test_set_list.txt".format(prefix), dtype=str)
            self.is_train = False             

        else:
            raise Exception('Variable "split" is not set properly!!')

        self.subject_list = sorted(self.subject_list.tolist())

        self.root = "{0}/render_THuman_with_blender/buffer_fixed_full_mesh".format(prefix)
        self.smplx_folder = "{0}/render_THuman_with_blender/buffer_fixed_smplx".format(prefix)

        self.smplx_blendweight_folder = "{0}/blendweights_smplx_rendering/RGB_maps_from_blendweightMaps_barycentric".format(prefix)


        self.img_files = []

        if self.justFrontal:
            with open('{0}/split_mesh/all_subjects_with_angles.txt'.format(prefix), 'r') as f:
                file_list = f.readlines()
                                
                # Need to add 270 degree to the frontal angle so that the frontal angle would be image_cond_270
                temp_list = []
                for line in file_list:
                    subject = line.split()[0]
                    frontal_angle = line.split()[1].replace('\n', '')
                    frontal_angle = int(frontal_angle) + 270

                    # converts angles that are more or equal to 360 degree
                    if frontal_angle>=360:
                        frontal_angle = frontal_angle - 360

                    # converts angles that are less 0 degree
                    if frontal_angle<0:
                        frontal_angle = frontal_angle + 360       

                    temp_list.append( (subject, frontal_angle) )
                file_list = temp_list

            for subject, frontal_angle in file_list:
                if subject in self.subject_list:
                    subject_render_filepath = os.path.join(self.root, subject, 'rendered_image_{0:03d}.png'.format( int(frontal_angle) ) )
                    self.img_files.append(subject_render_filepath)
        else:
            for subject in self.subject_list:
                subject_render_folder = os.path.join(self.root, subject)
                subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
                self.img_files = self.img_files + subject_render_paths_list

        self.img_files = sorted(self.img_files)

        self.train_index_for_test = 0


    def __len__(self):
        return len(self.img_files)

    def set_is_train(self, train_flag):
        self.is_train = train_flag

    def get_is_train(self):
        return self.is_train

    def get_sample_for_test(self):
        curr_index = self.train_index_for_test

        if self.justFrontal:
            self.train_index_for_test += 1
        else:
            self.train_index_for_test += 36 # since each subject has 36 images.

        return self.__getitem__( curr_index )

    def __getitem__(self, i):

        if self.is_train:
            num_of_prior_imgs = np.random.randint(low=1,high=4) # random int from [1-3]
        else: # i.e. we are not currently training
            num_of_prior_imgs = 3

        num_of_zero_padding_imgs = 3 - num_of_prior_imgs

        render_path = self.img_files[i]

        subject = render_path.split('/')[-2]

        yaw = render_path.split("_")[-1] # e.g. '045.png'
        yaw = yaw.replace('.png','')
        yaw = int(yaw)

        if not self.useBlendweights:
            smplx_image_path = os.path.join( self.smplx_folder, subject, 'rendered_SmplxImage_{0:03d}.png'.format(yaw) )
            smplx_image = Image.open(smplx_image_path).convert('RGB')
            smplx_image = np.array(smplx_image).astype(np.uint8) # (1024, 1024, 3)

            smplx_image = Image.fromarray(smplx_image, 'RGB')  
            smplx_image = smplx_image.resize( (self.img_size,self.img_size) )  
            smplx_image = np.array(smplx_image).astype(np.uint8) # (self.img_size, self.img_size, 3)

            smplx_image = (smplx_image/127.5 - 1.0).astype(np.float32)
        else:
            smplx_image = []
            for set_index in range(NUM_OF_BLENDWEIGHT_MAPS_TO_USE):
                smplx_blendweight_path = os.path.join( self.smplx_blendweight_folder, subject, "RGB_map_subject_{0}_angle_{1:03d}_{2}.png".format(subject, yaw, set_index) )
                smplx_blendweight = Image.open(smplx_blendweight_path).convert('RGB')
                smplx_blendweight = np.array(smplx_blendweight).astype(np.uint8) # (512, 512, 3)
                smplx_blendweight = (smplx_blendweight/127.5 - 1.0).astype(np.float32)
                smplx_image.append(smplx_blendweight)
            smplx_image = np.concatenate(smplx_image,axis=-1) # (512,512, 3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE)


        image_list = []
        num_of_prior_imgs_remaining = num_of_prior_imgs

        if (not self.useSequentialSmplx) and self.is_train:
            smplx_image_list = [0,0,0]
        else:
            smplx_image_list = [] # contain all smplx images except the smplx image in the target pose (i.e. 'smplx_image')
        
        for i in range(4):

            if num_of_prior_imgs_remaining == 0:
                zero_pad_img = np.zeros( [self.img_size, self.img_size, 3] ).astype(np.float32) # (self.img_size, self.img_size, 3)
                image_list.append(zero_pad_img)

                if (i >= 1) and ( (self.useSequentialSmplx) or (not self.is_train) ) :
                    if not self.useBlendweights:
                        zero_pad_img = np.zeros( [self.img_size, self.img_size, 3] ).astype(np.float32) # (self.img_size, self.img_size, 3)
                    else:
                        zero_pad_img = np.zeros( [self.img_size, self.img_size, 3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE] ).astype(np.float32) # (self.img_size, self.img_size, 3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE)
                    smplx_image_list.append(zero_pad_img)

                continue

            current_yaw = yaw - i * 90

            # converts angles that are more or equal to 360 degree
            if current_yaw>=360:
                current_yaw = current_yaw - 360

            # converts angles that are less 0 degree
            if current_yaw<0:
                current_yaw = current_yaw + 360       

            current_yaw = "{0:03d}".format(current_yaw)

            current_render_path = render_path.replace( "{0:03d}.png".format(yaw) , current_yaw + ".png" )

            mask_path = current_render_path.replace("_image_","_mask_")

            image = Image.open(current_render_path).convert('RGB')
            image = np.array(image).astype(np.uint8) # (1024, 1024, 3)
            
            mask = Image.open(mask_path).convert('L') # convert to grayscale (it shd already be grayscale)
            mask = np.array(mask).astype(np.uint8)[..., None] # (1024, 1024, 1)
            mask[mask!=0] = 1

            image = image * mask 

            image = Image.fromarray(image, 'RGB')  
            image = image.resize( (self.img_size,self.img_size) )  
            image = np.array(image).astype(np.uint8) # (self.img_size, self.img_size, 3)

            image = (image/127.5 - 1.0).astype(np.float32)
            image_list.append(image)

            if (not self.useSequentialSmplx) and self.is_train:
                pass 
            elif (i >= 1) and ( (self.useSequentialSmplx) or (not self.is_train) ) :

                if not self.useBlendweights:
                    current_smplx_image_path = os.path.join( self.smplx_folder, subject, 'rendered_SmplxImage_{0}.png'.format(current_yaw) )
                    current_smplx_image = Image.open(current_smplx_image_path).convert('RGB')
                    current_smplx_image = np.array(current_smplx_image).astype(np.uint8) # (1024, 1024, 3)

                    current_smplx_image = Image.fromarray(current_smplx_image, 'RGB')  
                    current_smplx_image = current_smplx_image.resize( (self.img_size,self.img_size) )  
                    current_smplx_image = np.array(current_smplx_image).astype(np.uint8) # (self.img_size, self.img_size, 3)

                    current_smplx_image = (current_smplx_image/127.5 - 1.0).astype(np.float32)
                else: 
                    current_smplx_image = []
                    for set_index in range(NUM_OF_BLENDWEIGHT_MAPS_TO_USE):
                        current_smplx_blendweight_path = os.path.join( self.smplx_blendweight_folder, subject, "RGB_map_subject_{0}_angle_{1}_{2}.png".format(subject, current_yaw, set_index) )
                        current_smplx_blendweight = Image.open(current_smplx_blendweight_path).convert('RGB')
                        current_smplx_blendweight = np.array(current_smplx_blendweight).astype(np.uint8) # (512, 512, 3)
                        current_smplx_blendweight = (current_smplx_blendweight/127.5 - 1.0).astype(np.float32)
                        current_smplx_image.append(current_smplx_blendweight)
                    current_smplx_image = np.concatenate(current_smplx_image,axis=-1) # (512,512, 3*NUM_OF_BLENDWEIGHT_MAPS_TO_USE)


                    
                smplx_image_list.append(current_smplx_image)


            if i == 0:
                pass 
            else:
                num_of_prior_imgs_remaining = num_of_prior_imgs_remaining - 1 


        example = {}
        example["image_target"] = image_list[0] # (self.img_size, self.img_size, 3) # is the groundtruth that we are trying to create
        example["image_cond"] = image_list[1] # (self.img_size, self.img_size, 3) # is the 90 degree prior
        example["image_cond_180"] = image_list[2]  # (self.img_size, self.img_size, 3) # is the 180 degree prior
        example["image_cond_270"] = image_list[3]  # (self.img_size, self.img_size, 3) # is the 270 degree prior

        example["num_of_prior_imgs"] = num_of_prior_imgs # is an integer
        example["smplx_image"] = smplx_image # (self.img_size, self.img_size, 3) # is the priors (will be a smplx image of the target image's pose)
        
        example["smplx_image_90"] = smplx_image_list[0]
        example["smplx_image_180"] = smplx_image_list[1]
        example["smplx_image_270"] = smplx_image_list[2]

        return example


class ThumanDatasetNovelTrain(ThumanDatasetNovel):
    def __init__(self, size, useBlendweights=False, useSequentialSmplx=False, justFrontal=False ):
        super().__init__(size, split='train', useBlendweights=useBlendweights, useSequentialSmplx=useSequentialSmplx, justFrontal=justFrontal )



class ThumanDatasetNovelValidation(ThumanDatasetNovel):
    def __init__(self, size, useBlendweights=False, useSequentialSmplx=False, justFrontal=False ):
        super().__init__(size, split='val', useBlendweights=useBlendweights, useSequentialSmplx=useSequentialSmplx, justFrontal=justFrontal )



class ThumanDatasetNovelTest(ThumanDatasetNovel):
    def __init__(self, size, useBlendweights=False, useSequentialSmplx=False, justFrontal=False ):
        super().__init__(size, split='test', useBlendweights=useBlendweights, useSequentialSmplx=useSequentialSmplx, justFrontal=justFrontal )







