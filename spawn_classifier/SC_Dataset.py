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

# Inherets Dataset and overwrites len and getitem
class SC_Dataset(Dataset):
    def __init__(self, title="main", data_path='data/', transform=None):
        self.title = title
        self.data_path = data_path
        self.data = np.loadtxt(f'{data_path}{title}.txt')
        self.video_cache = None

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    # must return the length of the data as int
    def __len__(self):
        return self.data.shape[0]

    # must return the image and class in (image_tensor, label_tensor) tuple
    def __getitem__(self, index):
        video, frame, label = self.data[index].astype(np.int32)

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

        frame_data = frames[frame]

        # print(f'Frame_data shape: {frame_data.shape}')
        # print(f'type of frame_data: {type(frame_data)}')

        if frame_data is None:
            raise ValueError(f'Video:{video}[{frame}] is None')

        return (self.transform(frame_data.copy()), torch.tensor(label, dtype=torch.long))