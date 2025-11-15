from processing_tools.data_generation import SpawnDataGen
from spawn_classifier.SC_Dataset import SC_Dataset
from torchvision import transforms
import numpy as np
import os, sys
from pathlib import Path
import cv2

pwd_to_raw = "data/raw_video"
pwd = "data/frame_data"

labelVideo = False

videos = [
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15"
]

label_videos = [
    "2",
    "3",
    "4",
    "5"
]

if labelVideo:
    for vid in videos:
        fname = f'{pwd_to_raw}/{vid}.mp4'
        gen = SpawnDataGen(fname)
        gen.smartfill(f'{pwd}/{vid}.txt', 70, 0, fill_0_to_0=True)

with open(f'main.txt', 'w') as mf:
    for vid in label_videos:
        fname = f'{pwd}/{vid}.txt'
        file_arr = np.loadtxt(fname).astype(np.int64)
        for frame, label in file_arr:
            mf.write(f'{vid} {frame} {label}\n')

t = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

dataset = SC_Dataset(f'main.txt', transform=t)
print(len(dataset))
image, label = dataset[0]


print(label)

