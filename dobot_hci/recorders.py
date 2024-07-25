import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.utils.plotting import Annotator
from cv2 import aruco
from .camera_feeds import RealSenseFeedProcessor
# from .models.vision import Florence2FT
from ultralytics import YOLO

from .settings import settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = YOLO(settings.CUSTOM_MODELS_DIR / "yolo" / "best.pt")  # load a pretrained model (recommended for training)


# model = YOLO(settings.CUSTOM_MODELS_DIR / "yolo" / "yolov8s.pt")  # load a pretrained model (recommended for training)


class ObjectRecorder:
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, **kwargs):
        self.camera_feed_proc = RealSenseFeedProcessor()
        # self.vision_model = Florence2FT()
        self.aurco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        self.aurco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aurco_dict, self.aurco_params)

    def _lazy_initialization(self):
        ...

    def run(self):
        def key_event_hook(e):
            ...

        def process_frame_hook(**kwargs):
            frame = kwargs["frame"]
            frame_gray = kwargs["frame_gray"]
            frame_hsv = kwargs["frame_hsv"]
            frame_rgb = kwargs["frame_rgb"]
            frame_shape = frame.shape
            h, w, c = frame_shape

            # Process the image with MediaPipe
            # results = self.pose.process(frame_rgb)
            # pil_image = Image.fromarray(frame_rgb)
            # # print(self.vision_model.infer(pil_image))
            # results = model.predict(frame_rgb, conf=0.01)
            #
            # # Process the results and draw bounding boxes
            # for result in results:
            #     boxes = result.boxes
            #     for box in boxes:
            #         # Get box coordinates
            #         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            #
            #         # Get class and confidence
            #         class_id = int(box.cls[0])
            #         conf = float(box.conf[0])
            #         class_name = model.names[class_id]
            #
            #         # Draw bounding box
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #
            #         # Prepare label
            #         label = f"{class_name} {conf:.2f}"
            #
            #         # Get text size
            #         (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            #
            #         # Draw filled rectangle for text background
            #         cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            #
            #         # Put text
            #         cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            corners, ids, rejected = self.detector.detectMarkers(frame_gray)

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)

            # Define range for orange color in RGB
            # lower_orange = np.array([0, 70, 0])
            # upper_orange = np.array([0, 255, 100])
            #
            # # Create a mask using color thresholding
            # mask = cv2.inRange(frame_rgb, lower_orange, upper_orange)
            #
            # result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)

            # cv2.imshow('YOLO Detections', frame)
            cv2.imshow('YOLO Detections', frame)
            # cv2.imshow('YOLO Detections3', result)

            # time.sleep(5 * 50)

            # np.savetxt("test.npy", frame_rgb)
            # sys.exit(0)

            # cv2.imshow('YOLO Detections', frame)

            # for result in results:
            #     boxes = result.boxes  # Bounding box object for each detection
            #     for box in boxes:
            #         class_id = int(box.cls[0])  # Get class ID
            #         class_name = model.names[class_id]  # Get class name
            #         bounding_box = box.xyxy[0].tolist()  # Get bounding box coordinates [x1, y1, x2, y2]
            #
            #         # Print in the requested format
            #         print(f"{{{class_name}: {bounding_box}}}")

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
