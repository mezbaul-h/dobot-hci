import time
from multiprocessing import Event
from typing import Optional, Generator
import cv2
import numpy as np


class CameraFeed:
    def __init__(self, **kwargs):
        self.shutdown_event: Optional[Event] = kwargs.pop('shutdown_event')
        self.camera_index = kwargs.get('camera_index', 0)
        self.cap: Optional[cv2.VideoCapture] = None
        self.target_fps = kwargs.get('target_fps', 30)
        self.frame_time = 1.0 / self.target_fps
        self.window_name = 'Camera Feed'
        self.fps = None
        self.new_frame_time = None
        self.prev_frame_time = None
        self.last_fps_update = None
        self.frame_update_interval = 2  # seconds

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        drawn_frame = frame.copy()
        height, width = drawn_frame.shape[:2]

        # Draw a rectangle
        # cv2.rectangle(drawn_frame, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), (0, 255, 0), 2)

        # Draw FPS in upper right corner with glass effect
        fps_text = f"{fps:.1f} FPS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]

        # Background rectangle
        bg_rect_width = text_size[0] + 20
        bg_rect_height = text_size[1] + 20
        bg_rect_top_left = (width - bg_rect_width - 10, 10)
        bg_rect_bottom_right = (width - 10, bg_rect_height + 10)

        # Draw semi-transparent background
        overlay = drawn_frame.copy()
        cv2.rectangle(overlay, bg_rect_top_left, bg_rect_bottom_right, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, drawn_frame, 0.5, 0, drawn_frame)

        # Draw text with glass effect
        text_pos = (width - text_size[0] - 20, bg_rect_height - 5)
        cv2.putText(drawn_frame, fps_text, text_pos, font, font_scale, (200, 200, 200), font_thickness + 3, cv2.LINE_AA)
        cv2.putText(drawn_frame, fps_text, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return drawn_frame

    def update_fps(self, frame: np.ndarray):
        if self.prev_frame_time:
            self.fps = 1 / (self.new_frame_time - self.prev_frame_time)
        else:
            height, width = frame.shape[:2]
            cv2.resizeWindow(self.window_name, width, height)
            self.fps = 0

        self.last_fps_update = time.time()  # Update timestamp

    def run(self):
        if not self.cap:
            raise RuntimeError("Camera is not initialized. Use 'with' statement or call __enter__().")

        # Create a named window with the desired properties
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()

            if not ret or (self.shutdown_event and self.shutdown_event.is_set()):
                break

            self.new_frame_time = time.time()

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Update FPS every 3 seconds
            if self.last_fps_update is None or (time.time() - self.last_fps_update) >= self.frame_update_interval:
                self.update_fps(frame)

            self.prev_frame_time = self.new_frame_time

            processed_frame = self.draw_fps(frame.copy(), self.fps)

            cv2.imshow(self.window_name, processed_frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            cv2.waitKey(1)
