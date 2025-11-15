# Timothy Panilaitis
# 10/31/2025
# Spawn Classifier Feature Pipeline (SCF)
# Contains all methods needed to translate initial feature array into 
# most informative features. 

# Most methods expect the Imagetensor

import sys
import numpy as np
import skimage as ski
import torchvision
import torch
import cv2

class SCF_Pipeline:
    def __init__(self, pipeline_choice):
        
        v2.compose()

    def run():
        pass
    
    def master_pipeline(self, frame_data, augmentation_arr, aug_param_arr = None):
        if not isinstance(frame_data, torch.Tensor) or not augmentation_arr:
            sys.stderr(f'ERROR: feature array is not Tensor or augmentation array is None')
        
        aug_feature_arr = frame_data
        additions = []
        for aug in augmentation_arr:
            aug, add = aug(aug_feature_arr)
            if aug:
                aug_feature_arr = aug
            if add:
                additions.append(add)

        return np.flatten([aug_feature_arr, np.array(additions)]) 
    
    def reduce_image_quality(self, feature_arr, perc = 0.5):
        return (ski.transform.rescale(feature_arr, 0.5), None)

    def remove_color_channel(self, feature_arr, color = None):
        if color is None:
            return feature_arr
        
        color = color.lower()
        if color == "red":
            color_num = 1
        elif color == "green":
            color_num = 2
        elif color == "blue":
            color_num = 3
        else:
            sys.stderr(f'ERROR: color:{color} is not of type "Red", "Green", and "Blue"')

        # simple np remove
    
    def convert_grayscale(self, feature_arr):
        return (ski.color.rgb2gray(feature_arr), None)
    
    def shape_recognition(self, feature_arr):
        return
    
    def color_hist(self, feature_arr, nbins=15, value_range=(0,256)):
        bin_edges = np.linspace(value_range[0], value_range[1], nbins + 1)
        hists = [np.histogram(feature_arr[..., c], bins=bin_edges)[0] for c in range(3)]
        hists = np.array(hists)
        
        return (None, hists.ravel())
         

            
