import typing
from contextlib import AbstractContextManager
from pathlib import Path

import mediapipe as mp
import numpy as np
from savant.networks.cnn import CNNRunner
from savant.utils import pad_numpy_array
from wisdom.utils import get_bounding_box, landmark_to_numpy, landmark_to_ratio

from .settings import BODY_POSE_LANDMARK_COUNT


class RealTimeLandmarkInference(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._initialize()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, **kwargs):
        self.camera_feed_proc = kwargs["camera_feed_processor_class"]()
        self.network_checkpoint_filename = kwargs["network_checkpoint_filename"]
        self.network: typing.Optional[CNNRunner] = None

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65)

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

    def _initialize_network(self):
        runner = CNNRunner()

        if not (Path(self.network_checkpoint_filename).exists() and Path(self.network_checkpoint_filename).is_file()):
            raise RuntimeError(f"Checkpoint file not found: {self.network_checkpoint_filename}")

        runner.load_state(self.network_checkpoint_filename)

        self.network = runner

    def _initialize(self):
        self._initialize_network()

    def run(self):
        def process_frame_hook(**kwargs):
            frame = kwargs["frame"]
            frame_rgb = kwargs["frame_rgb"]
            h, w, c = frame.shape
            predicted_pose = False

            # Process the image with MediaPipe for pose.
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                body_landmarks = results.pose_landmarks
                # for body_landmarks in results.pose_landmarks:
                bbox = get_bounding_box((w, h), body_landmarks)

                processed_landmark_data = []
                pose_landmark_indexes = []

                # print(body_landmarks.landmark)
                for idx, landmark in enumerate(body_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    if landmark.visibility > 0.5:
                        pose_landmark_indexes.append(idx)

                    # Draw landmarks on image
                    # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                    # Convert landmark to ratio
                    ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                    processed_landmark_data.append((ratio_x, ratio_y))

                # Infer (predict gesture).
                if processed_landmark_data and len(set(list(range(25))) - set(pose_landmark_indexes)) == 0:
                    # Draw the landmarks and a rectangle around it.
                    self.camera_feed_proc.draw_rectangle(frame, bbox)
                    self.mp_drawing.draw_landmarks(frame, body_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    label_probabilities = self.network.predict(np.array([processed_landmark_data]))
                    label_with_highest_probability = next(iter(label_probabilities))

                    self.camera_feed_proc.draw_text_around_bounding_box(
                        frame,
                        bbox,
                        f"{label_with_highest_probability} "
                        f"({label_probabilities[label_with_highest_probability]:.2f})",
                    )

                    predicted_pose = True

            if not predicted_pose:
                # Process the image with MediaPipe for hand gestures.
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        bbox = get_bounding_box((w, h), hand_landmarks)

                        processed_landmark_data = []

                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x, y = int(landmark.x * w), int(landmark.y * h)

                            # Convert landmark to ratio
                            ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                            processed_landmark_data.append((ratio_x, ratio_y))

                        # Infer (predict gesture).
                        if processed_landmark_data:
                            self.camera_feed_proc.draw_rectangle(frame, bbox)
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                            padded_landmark_data = pad_numpy_array(
                                np.array(processed_landmark_data), (BODY_POSE_LANDMARK_COUNT, 2)
                            )
                            label_probabilities = self.network.predict(np.array([padded_landmark_data]))
                            label_with_highest_probability = next(iter(label_probabilities))

                            self.camera_feed_proc.draw_text_around_bounding_box(
                                frame,
                                bbox,
                                f"{label_with_highest_probability} "
                                f"({label_probabilities[label_with_highest_probability]:.2f})",
                            )

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
