# Timothy Panilaitis
# 10/31/2025
# Spawn Classifier Dataset (SCD)
# Class created to interact with the tensor dataloader

from processing_tools.Video import Video
from torch.utils.data import Dataset
from torchvision import transforms

import torch
from collections import OrderedDict
import numpy as np
import sys

class LRUCache():
    def __init__(self, capacity=12):
        self.capacity = capacity
        self.data = OrderedDict()

    # key by video
    def get(self, key):
        try:
            val = self.data.pop(key)
            self.data[key] = val
            return val
        except KeyError:
            return None
    
    def put(self, key, value):
        if key in self.data:
            self.data.pop(key)
        elif len(self.data) >= self.capacity:
            self.data.popitem(last=False)
        self.data[key] = value

# Inherets Dataset and overwrites len and getitem
class SC_Dataset(Dataset):
    def __init__(self, data_dir, rel_path='data/raw_video/', transform=None):
        self.data_dir = data_dir
        self.data = np.loadtxt(data_dir).astype(np.int64)
        self.video_cache = None
        self.rel_path = rel_path

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    # must return the length of the data as int
    def __len__(self):
        return self.data.shape[0]

    def _init_worker_cache(self):
        self.video_cache = LRUCache()

    # must return the image and class in (image_tensor, label_tensor) tuple
    def __getitem__(self, index):
        if self.video_cache is None:
            self._init_worker_cache()
        
        video, frame, label = self.data[index]

        cached = self.video_cache.get(video)
        if cached is not None:
            video = cached
        else:
            video = Video(f'{self.rel_path}{video}.mp4')
            self.video_cache.put(video, video)
        
        frame_data = video.at(frame)
        if frame_data is None:
            raise ValueError(f'Video:{video}[{frame}] is None')

        return (self.transform(frame_data), torch.tensor(label, dtype=torch.long))