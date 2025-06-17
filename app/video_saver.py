import cv2

class VideoSaver:
    def __init__(self, filename, width=640, height=480, fps=20.0):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
