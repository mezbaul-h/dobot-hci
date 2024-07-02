import time
from contextlib import AbstractContextManager

import mediapipe as mp
import numpy as np

from .settings import TRAINING_DATA_DIR
from .utils import get_bounding_box, landmark_to_ratio


class RealTimeHandGestureRecorder(AbstractContextManager):
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
            frame_shape = frame.shape
            h, w, c = frame_shape

            # Process the image with MediaPipe
            results = self.hands.process(frame_rgb)

            # This will only contain the final landmark data if there are more.
            processed_landmark_data = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    bbox = get_bounding_box((w, h), hand_landmarks)

                    # Draw the landmarks and a rectangle around it.
                    self.camera_feed_proc.draw_rectangle(frame, bbox)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    processed_landmark_data = []

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)

                        # Draw landmarks on image
                        # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                        # Convert landmark to ratio
                        ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                        processed_landmark_data.append((ratio_x, ratio_y))

                    self.record_landmark(processed_landmark_data)

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)


class RealTimeBodyPoseRecorder(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, **kwargs):
        self.label = kwargs["label"]

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
        def key_event_hook(e):
            ...

        def process_frame_hook(**kwargs):
            frame = kwargs["frame"]
            frame_rgb = kwargs["frame_rgb"]
            frame_shape = frame.shape
            h, w, c = frame_shape

            # Process the image with MediaPipe
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                body_landmarks = results.pose_landmarks
                # for body_landmarks in results.pose_landmarks:
                bbox = get_bounding_box((w, h), body_landmarks)

                # Draw the landmarks and a rectangle around it.
                self.camera_feed_proc.draw_rectangle(frame, bbox)
                self.mp_drawing.draw_landmarks(frame, body_landmarks, self.mp_pose.POSE_CONNECTIONS)

                processed_landmark_data = []

                for idx, landmark in enumerate(body_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    # Draw landmarks on image
                    # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                    # Convert landmark to ratio
                    ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                    processed_landmark_data.append((ratio_x, ratio_y))

                self.record_landmark(processed_landmark_data)

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
