from spawn_detection.data_generation import SpawnDataGen

pwd_to_raw = "data/raw_video/"
pwd = "../data/frame_data"

videos = [
    "03",
    "04"
]
for video in videos:
    frame_skip = 60
    print(f'{pwd_to_raw}{video}.mp4')
    item = SpawnDataGen(f'{pwd_to_raw}{video}.mp4')
    item.smartfill(f'{video}_filled.txt', frame_skip, fill_0_to_0=True)