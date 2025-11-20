import pytest
import torch
import numpy as np
from spawn_classifier.SC_Dataset import SC_Dataset
from mocks.video_mock import VideoMock

# need to mock json return and memmap.
@pytest.fixture
def dataset(tmp_path):    
    ann_file = tmp_path / "test_data.txt"
    
    col0 = np.random.randint(1, 6, size=100)
    col1 = np.arange(100)
    col2 = np.random.randint(0, 2, size=100)
    arr = np.column_stack((col0, col1, col2))
    np.savetxt(ann_file, arr)

    yield SC_Dataset(title="test_data", rel_path=tmp_path)

def test_get_item(dataset):
    image, label = dataset.__getitem__(0)
    print(type(image))
    assert(isinstance(image, torch.Tensor))
    assert(isinstance(label, torch.Tensor))
