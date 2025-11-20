import cv2
import numpy as np

def setup_module():
    width, height = 64, 64
    num_frames = 80
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # lossless
    out = cv2.VideoWriter("test_video.avi", fourcc, fps, (width, height))

    frames = []
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i, dtype=np.uint8)
        out.write(frame)
        frames.append(frame)

    out.release()
    np.save("test_video_frames.npy", np.array(frames))

setup_module()
