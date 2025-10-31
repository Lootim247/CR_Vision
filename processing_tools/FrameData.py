# Timothy Panilaitis
# 10/27/2025
# VideoToImage class
# supports indexing from mp4 to np array of shape
# [frames, color_channels, height, width]

import cv2
import numpy as np

class FrameData:
    def __init__(self):
        self.master = None
    
    def __del__(self):
        cv2.destroyAllWindows()

    def vid_to_frames(self, video, fps=None):
        cap = cv2.VideoCapture(video)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_arr = []
        if fps:
            video_frac = video_fps / fps
        else:
            video_frac = 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if int(frame_count % video_frac) == 0:
                frame_arr.append(frame)
            
            frame_count += 1        

        self.master = np.array(frame_arr).transpose(0, 3, 1, 2)
        cap.release()
    
    def vid_to_file(self, video, file, fps=None):
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        channels = 3
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        skip = max(int(video_fps / fps), 1) if fps else 1
        n_saved_frames = (frame_count + skip - 1) // skip  # ceil division
        
        frames_memmap = np.lib.format.open_memmap(
            file,
            mode='w+',
            dtype='uint8',
            shape=(n_saved_frames, height, width, channels)
        )

        count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % skip == 0:
                frames_memmap[saved_count] = frame
                saved_count += 1
            
            count += 1

        cap.release()

        
    def write(self, file):
        self.master = np.save(file, self.master)

    def read(self, file):
        self.master = np.load(file)