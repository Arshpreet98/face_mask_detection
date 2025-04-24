import cv2
import time
from threading import Thread
from collections import deque

class VideoStream:
    """
    Class for efficient video streaming using threading
    """
    def __init__(self, src=0, name="VideoStream"):
        self.stream = cv2.VideoCapture(src)
        self.name = name
        self.stopped = False
        self.frame = None
        self.grabbed = False
        (self.grabbed, self.frame) = self.stream.read()
        self.fps_deque = deque(maxlen=30)  # Store last 30 frame timestamps for FPS calculation

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                continue
            (self.grabbed, self.frame) = self.stream.read()
            self.fps_deque.append(time.time())

    def read(self):
        return self.frame

    def get_fps(self):
        if len(self.fps_deque) <= 1:
            return 0
        return len(self.fps_deque) / (self.fps_deque[-1] - self.fps_deque[0])

    def stop(self):
        self.stopped = True
        self.stream.release()