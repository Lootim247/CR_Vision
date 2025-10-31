# Timothy Panilaitis
# 10/30/2025
# enables generation of a binary dataset off a video using different labeling
# functions. smart fill enables an easier filling experience.
import numpy as np
import cv2
from processing_tools.Video import Video

class SpawnDataGen:
    def __init__(self, video):
        self.video = Video(video)

    def generate_data(self, fname, frame_diff, frame_start=0):
        """
        Manually label frames in the video. Use keys:
        - '0' to label frame as 0
        - '1' to label frame as 1
        - '/' to skip
        - 'b' to undo last label
        """
        labeled_frames = []
        curr_frame = frame_start

        while True:
            frame = self.video.at(curr_frame)
            if frame is None:
                break

            label = self.get_label_for_frame(curr_frame)
            if label == "back":
                if labeled_frames:
                    last_frame, _ = labeled_frames.pop()
                    curr_frame = last_frame - frame_diff
                    continue
            elif label is not None:
                labeled_frames.append([curr_frame, label])

            curr_frame += frame_diff

        cv2.destroyAllWindows()
        if labeled_frames:
            # Save as numeric array to avoid pickling issues
            np.savetxt(fname, np.array(labeled_frames, dtype=int), fmt="%d")

    def get_label_for_frame(self, frame_index):
        """
        Display the frame and get a label from the user.
        Returns:
            0, 1, None (skip), or "back"
        """
        frame = self.video.at(frame_index)
        if frame is None:
            return None

        scale_factor = 0.3
        h, w = frame.shape[:2]
        display_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("0"):
            return 0
        elif key == ord("1"):
            return 1
        elif key == ord("/"):
            return None
        elif key == ord("b"):
            return "back"
        else:
            return None

    def smartfill(self, fname, frame_diff, frame_start=0, fill_1_to_1=False, fill_0_to_0=True):
        """
        Automatically fills gaps between labeled frames using recursive midpoint labeling.
        """
        self.generate_data(fname, frame_diff, frame_start)

        if not fill_0_to_0 and not fill_1_to_1:
            return

        # Load labeled frames as numeric array
        file_arr = np.loadtxt(fname, dtype=int)
        if file_arr.size == 0:
            return

        max_frame = int(file_arr[-1, 0])
        min_frame = int(file_arr[0, 0])
        arr = np.full((max_frame - min_frame + 1,), -1, dtype=int)
        for f, val in file_arr:
            arr[int(f - min_frame)] = int(val)

        def recursive_fill(start, end):
            if end - start <= 1:
                return

            # Check boundaries
            start_val = arr[start]
            end_val = arr[end]
            print(f'Start-End: {start_val}-{end_val}')

            # Auto-fill if both sides match and corresponding fill flag is True
            if start_val == end_val and start_val != -1:
                if (start_val == 0 and fill_0_to_0) or (start_val == 1 and fill_1_to_1) or (end - start - 2 <= 2):            
                    arr[start+1:end-1] = start_val  # fill interior
                    print(f'Filling from {start + 1}:{end} with {start_val}s')
                    return

            # Otherwise, use midpoint labeling
            mid = (start + end) // 2
            if arr[mid] == -1:
                print(f'Checking mid index:{mid}')
                label = self.get_label_for_frame(min_frame + mid)
                print(f'arr[{mid}] = {label}')
                if label is not None and label != "back":
                    arr[mid] = label

            # Recurse on left and right subsegments
            if start < mid:
                print(f'sub-recursion from {start}:{mid}')
                recursive_fill(start, mid)
            if mid < end:
                print(f'sub-recursion from {mid}:{end}')
                recursive_fill(mid, end)


        # Fill recursively between existing labeled frames (interior frames only)
        for i in range(arr.shape[0]):
            if arr[i] == -1:
                j = 0
                while arr[i+j] == -1:
                    j += 1
                print(f'recursively filling {i-1}:{i+j-1}')
                recursive_fill(i - 1, i + j)
                

        # Save filled frames
        filled_frames = np.array([(i + min_frame, val) for i, val in enumerate(arr)], dtype=int)
        np.savetxt(fname.replace(".txt", "_smart_filled.txt"), filled_frames, fmt="%d")
