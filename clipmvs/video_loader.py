import cv2
from PIL import Image

class VideoDataLoader:
    """
    A class to load video frames in batches and fetch specific frames by timestamp.
    """
    def __init__(self, video_path, batch_size=1, start_frame=0, end_frame=None, interval=1):
        """
        Initialize the VideoDataLoader.

        Args:
            video_path (str): Path to the video file.
            batch_size (int): Number of frames to fetch in each batch.
            start_frame (int): The starting frame of the video.
            end_frame (int): The ending frame of the video.
            interval (int): Interval between frames to fetch.
        """
        self.video_path = video_path
        self.batch_size = batch_size
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.interval = interval
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video stream or file")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.end_frame is None:
            self.end_frame = self.total_frames
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
    
    def __iter__(self):
        self.current_frame = self.start_frame
        return self
    
    def __next__(self):
        if self.current_frame >= self.end_frame:
            self.cap.release()
            raise StopIteration
        
        frames = []
        timestamps = []
        for _ in range(self.batch_size):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret or self.current_frame >= self.end_frame:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
            
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamps.append(timestamp)
            
            self.current_frame += self.interval
        
        if len(frames) == 0:
            self.cap.release()
            raise StopIteration
        
        return frames, timestamps
    
    def __len__(self):
        return (self.end_frame - self.start_frame + self.interval - 1) // self.interval

    def get_frame_by_timestamp(self, timestamp):
        """
        Get a specific frame by timestamp.

        Args:
            timestamp (float): Timestamp in seconds.

        Returns:
            PIL.Image: Frame at the given timestamp.
        """
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            return pil_image
        else:
            return None
