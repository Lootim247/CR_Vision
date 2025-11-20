# Timothy Panilaitis
# 10/27/2025
# Video class
# supports indexing from mp4 to np array of shape using ffmpeg (more secure)
# [color_channels, height, width]

import numpy as np
import sys
import shlex
import subprocess

class Video:
    def __init__(self, video):
        cmd = (
            f'ffprobe -v error -select_streams v:0 '
            f'-show_entries stream=width,height '
            f'-of csv=p=0 "{video}"'
        )
        out = subprocess.check_output(shlex.split(cmd)).decode().strip()
        w, h = map(int, out.split(','))
        self.width = w
        self.height = h

        self.video = video

    def at(self, frame_index):
        # FFmpeg command that outputs JUST the raw pixel bytes
        cmd = (
            f'ffmpeg -i "{self.video}" -vf "select=eq(n\\,{frame_index})" '
            f'-vframes 1 -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -'
        )

        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        raw = proc.stdout.read()  # raw RGB bytes

        if len(raw) < self.width * self.height * 3:
            raise ValueError(f"Frame {frame_index} not found or ffmpeg failed")


        frame = np.frombuffer(raw, np.uint8)
        frame = frame.reshape((self.height, self.width, 3))  # RGB
        frame = np.transpose(frame, (2, 0, 1)).copy()

        return frame
