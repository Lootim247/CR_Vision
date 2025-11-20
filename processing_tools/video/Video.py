# Timothy Panilaitis
# 10/27/2025
# Video class
# supports indexing from mp4 to np array of shape
# [color_channels, height, width]

import cv2
import numpy as np
import sys

class Video:
    def __init__(self, video):
        self.video = video
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise ValueError(f'VideoCapture failed to open {video}')

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def at(self, frame_index):
        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, current_frame = self.cap.read()

        if ret:
            return current_frame
        else:
            return None