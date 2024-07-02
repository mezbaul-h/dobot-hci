import time
from contextlib import AbstractContextManager

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from .common import CameraFeedProcessor, RealSenseFeedProcessor
from .settings import TRAINING_DATA_DIR, YOLO_ROOT_DIR
from .utils import get_bounding_box, landmark_to_ratio

model = YOLO(YOLO_ROOT_DIR / "yolov8s.pt")  # load a pretrained model (recommended for training)


class RealTimeYoloRecorder(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, **kwargs):
        self.label = kwargs["label"]

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65)

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

        self.training_data_dir = TRAINING_DATA_DIR / self.__class__.__name__ / self.label

        self.camera_feed_proc = kwargs["camera_feed_processor_class"]()

    def _lazy_initialization(self):
        # Lazy initializations.
        if not self.training_data_dir.exists():
            self.training_data_dir.mkdir(parents=True)

    def record_landmark(self, landmark_data):
        timestamp = int(time.time())
        filename = f"{self.__class__.__name__}__{self.label}__{timestamp}.npy"
        np.save(self.training_data_dir / filename, np.array(landmark_data).flatten())

    def run(self):
        def process_frame_hook(**kwargs):
            frame = kwargs["frame"]
            frame_rgb = kwargs["frame_rgb"]
            depth_image = kwargs["depth_image"]

            # Process the image with MediaPipe
            # results = self.pose.process(frame_rgb)

            results = model.predict(frame_rgb, conf=0.5)

            annotator = Annotator(frame)
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                    center_x = int((b[0] + b[2]) // 2)
                    center_y = int((b[1] + b[3]) // 2)
                    distance = depth_image[center_y, center_x]
                    # print(f'Distance: {distance} mm')

                    c = box.cls

                    annotator.box_label(b, f"{model.names[int(c)]} D: {distance}")

            frame = annotator.result()

            # if results.pose_landmarks:
            #     body_landmarks = results.pose_landmarks
            #     # for body_landmarks in results.pose_landmarks:
            #     bbox = get_bounding_box((w, h), body_landmarks)
            #
            #     # Draw the landmarks and a rectangle around it.
            #     self.camera_feed_proc.draw_rectangle(frame, bbox)
            #     self.mp_drawing.draw_landmarks(frame, body_landmarks, self.mp_pose.POSE_CONNECTIONS)
            #
            #     processed_landmark_data = []
            #
            #     for idx, landmark in enumerate(body_landmarks.landmark):
            #         x, y = int(landmark.x * w), int(landmark.y * h)
            #
            #         # Draw landmarks on image
            #         # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            #
            #         # Convert landmark to ratio
            #         ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
            #         processed_landmark_data.append((ratio_x, ratio_y))

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
