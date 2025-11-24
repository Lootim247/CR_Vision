# Timothy Panilaitis
# 10/31/2025
# Spawn Classifier Dataset (SCD)
# Class created to interact with the tensor dataloader

from torch.utils.data import Dataset
from torchvision import transforms

import torch
import numpy as np
import sys
import json

class SC_Dataset(Dataset):
    """
    INPUTS:
        data_set:   Expected to be a numpy array of shape [SIZE, 3] with sub-
                    array of property [VIDEO FRAME LABEL]
        data_path:  Should be the relative path to the data directory. Data is
                    expected to have a subfolder named "frame_arrays"
        transform:  A composition of pytorch transform function. If NONE auto
                    converts to tensor. 
    """
    def __init__(self, data_set, data_path='data/', transform=None):
        self.data_path = data_path
        self.data = np.array(data_set)
        self.video_cache = None

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        """
        OUTPUTS:
            length: The number of datapoints in the provided dataset.
        """
        return self.data.shape[0]

    def __get_item_helper(self, video, frame, label):
        json_path = f'{self.data_path}frame_arrays/{video}_meta.json'
        with open(json_path, "r", encoding="utf-8") as f:
            self.json = json.load(f)
        
        frames = np.memmap(
            f'{self.data_path}frame_arrays/{video}.dat', 
            dtype=np.uint8, 
            mode="r", 
            shape=(self.json["num_frames"], 
            self.json["height"], 
            self.json["width"], 
            3))

        frame_data = np.array(frames[frame])
        if frame_data is None:
            raise ValueError(f'Video:{video}[{frame}] is None')
        
        return [self.transform(frame_data.copy()), torch.tensor(label, dtype=torch.long)]

    def __getitem__(self, index):
        """
        INPUTS:
            index:  The index or array of index of the desired item(s) in the 
                    provided dataset.
        OUTPUTS:
            (frame tensor, :    The respective frame(s) refered to at the given 
                                dataset index(es).
            label tensor ) :    The respective label(s) at the given dataset
                                index(es).
        """
        
        if np.isscalar(index):
            video, frame, label = self.data[index].astype(np.int32)
            return self.__get_item_helper(video, frame, label)
        else:
            if not isinstance(index, np.ndarray):
                index = np.array(index)
            
            data = []
            for j in range(index.shape[0]):
                video, frame, label = self.data[index[j]].astype(np.int32)
                data.append(self.__get_item_helper(video, frame, label))
            return np.array(data)