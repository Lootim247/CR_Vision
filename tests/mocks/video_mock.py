import numpy as np

class VideoMock:
    def __init__(self, v_str):
        pass

    def at(self, i):
        if i <= 1000 and i >= 0:
            np.random.seed(int(i))
            return np.random.randint(0, 256, size=(1280, 720, 3), dtype=np.uint8)
        else:
            return None