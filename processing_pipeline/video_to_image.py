import cv2

class VideoToImage:
    def __init__(self, video, pwd):
        self.video = video
        self.pwd = pwd
        self.cap = cv2.VideoCapture(video)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def convert_to_frames(self, fps=10):
        video_frac = self.video_fps / fps
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if int(frame_count % video_frac) == 0:
                cv2.imwrite(f'{self.pwd}/{self.video}/frame_{frame_count:04d}.jpg', frame)
            
            frame_count += 1