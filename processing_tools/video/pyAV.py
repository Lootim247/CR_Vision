# Timothy Panilaitis
# 10/31/2025
# Video class using PyAV for fast frame access
# Returns numpy array: [channels, height, width] (RGB)

import av
import numpy as np

class Video:
    def __init__(self, path):
        """
        Open a video using PyAV.
        """
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.width = self.stream.width
        self.height = self.stream.height
        self.num_frames = self.stream.frames  # total number of frames (may be 0 for some files)
    
    # Timothy Panilaitis
# 10/31/2025
# Video class using PyAV for fast random-access frame retrieval
# Returns numpy array: [channels, height, width] (RGB)

import av
import numpy as np

class Video:
    def __init__(self, path):
        """
        Open a video using PyAV.
        """
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.width = self.stream.width
        self.height = self.stream.height
        self.num_frames = self.stream.frames  # may be 0 for some files

    def at(self, frame_index: int) -> np.ndarray:
        self.container.seek(frame_index, stream=self.stream, any_frame=True)
        frame = next(self.container.decode(video=0))
        img = frame.to_ndarray(format="rgb24")
        return np.transpose(img, (2, 0, 1))


    def __getitem__(self, idx: int):
        return self.at(idx)

    def shape(self):
        return (3, self.height, self.width)

