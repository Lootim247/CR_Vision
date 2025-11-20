import pytest
import torch
import cv2
from pathlib import Path
import numpy as np
from processing_tools.video.pyAV import Video

@pytest.fixture
def frames_array():
    folder = Path(__file__).parent
    return np.load(folder / "test_video_frames.npy")

@pytest.fixture
def video(tmp_path):
    folder = Path(__file__).parent
    p = folder / "test_video.avi"
    yield Video(p)

def test_video_at(video):
    # easy and non indexed order
    assert(isinstance(video.at(0), np.ndarray))
    with pytest.raises(IndexError):
        video.at(10000000)

def test_ooo_at(video):
    # out of order seeking
    assert(isinstance(video.at(0), np.ndarray))
    assert(isinstance(video.at(60), np.ndarray))
    assert(isinstance(video.at(31), np.ndarray))

@pytest.mark.parametrize("frame_idx", range(0, 80))
def test_at_robust(frames_array, video, frame_idx):
    # transpose to match [C,H,W] format
    expected = frames_array[frame_idx].transpose(2,0,1)
    actual = video.at(frame_idx)
    
    # exact pixel equality, lossless
    np.testing.assert_allclose(actual.astype(np.float32),
                               expected.astype(np.float32), atol=5)
