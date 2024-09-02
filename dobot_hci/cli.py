# from .recorders import ObjectRecorder
import json
import logging
import math
import random
import re
import signal
import sys
import time
from multiprocessing import Event, Queue, Process, Manager
from typing import Optional, List

import click
from PIL import Image
from colorama import Fore
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from .camera import CameraFeed
from .models.language import OllamaLM
from .models.speech import SpeechRecognizer
from .utils import print_system_message, log_to_queue
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
import multiprocessing
import queue
import time
from .settings import settings
from .models.vision import MicrosoftFlorence2


_exit_pattern = re.compile(r"\b(exit|quit|stop)\b", re.IGNORECASE)
lm = OllamaLM()

_system_prompt = """
You are an AI assistant that generates JSON-formatted data for controlling a Dobot Magician robotic arm. Your task is to interpret user commands related to moving objects and output the necessary sequence of tool calls in raw JSON format.

You have access to four tools:

1. `move_to_object`: Find the position of and move to an object given its label ID.
3. `pick`: Activate the suction to pick up an object at the current position.
4. `drop`: Deactivate the suction to drop an object at the current position.


You must generate a sequence of JSON-formatted tool calls based on the user’s request. The response should be in the following format:
[
  {
    "tool": "move_to_object",
    "parameters": {
      "label_id": "<OBJECT_LABEL>"
    }
  },
  {
    "tool": "pick"
  },
  {
    "tool": "drop"
  }
]


Instructions:
- Label IDs: The available and allowed label IDs are `CELL_PHONE`, `MOUSE`, `PEN`, and `USER_HAND`.
- Handling Requests: If the object cannot be described with these labels, respond with {"error": "Invalid object label"}.
- Response Format: Always respond with a JSON object or array of tool calls as shown above.


Example Scenario: If the user asks you to pick up a pen and give it to them, the sequence might involve finding the pen, moving to its position, picking it up, moving to the user's hand position, and dropping it.

Example Request: "Pick up the cell phone and place it in my hand."

Expected JSON Output: [ { "tool": "move_to_object", "parameters": { "label_id": "CELL_PHONE" } }, { "tool": "pick" }, { "tool": "move_to_object", "parameters": { "label_id": "USER_HAND" } }, { "tool": "drop" } ]


If you cannot map any of the objects to one of the available labels or if the request is outside of your capabilities, respond with: {"error": "I'm sorry, I can only assist with moving objects using the Dobot arm."}

Your response must only be in raw JSON format as described above, with no additional text, explanations, or markdown notation wraps.
"""


class YOLOVisionModel:
    def __init__(self):
        self.model = YOLO('yolov8s-worldv2.pt', verbose=False)
        print(self.model.names)


class ImageSharer:
    def __init__(self, max_size=(1920, 1080, 3)):  # Adjust max_size as needed
        self.max_size = max_size
        self.shared_array = multiprocessing.Array('B', max_size[0] * max_size[1] * max_size[2])
        self.metadata = multiprocessing.Array('i', 3)  # To store current image dimensions

    def update_image(self, cv_img):
        h, w, c = cv_img.shape
        if h > self.max_size[0] or w > self.max_size[1] or c != self.max_size[2]:
            raise ValueError("Image size exceeds maximum allowed dimensions")

        # Update metadata
        self.metadata[0] = h
        self.metadata[1] = w
        self.metadata[2] = c

        # Update image data
        shared_np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8)
        shared_np_array[:h * w * c] = cv_img.ravel()

    def get_image(self):
        h, w, c = self.metadata
        shared_np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8)
        return shared_np_array[:h * w * c].reshape(h, w, c)


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(tuple)
    update_object_positions_signal = pyqtSignal(dict)

    def __init__(self, **kwargs):
        super().__init__()
        self._run_flag = True
        self.shutdown_event: Optional[Event] = kwargs.get('shutdown_event')
        self.camera_index = kwargs.get('camera_index', 0)
        self.object_positions = kwargs['object_positions']
        self.fps = None
        self.new_frame_time = None
        self.prev_frame_time = None
        self.last_fps_update = None
        self.frame_update_interval = 2  # seconds
        self.yolo_vision_model = YOLOVisionModel()
        self.aruco_detector = None
        self.aruco_last_detected_at = time.time()

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        drawn_frame = frame.copy()
        height, width = drawn_frame.shape[:2]

        # Draw a rectangle
        # cv2.rectangle(drawn_frame, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), (0, 255, 0), 2)

        # Draw FPS in upper right corner with glass effect
        fps_text = f"{fps:.1f} FPS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]

        # Background rectangle
        bg_rect_width = text_size[0] + 20
        bg_rect_height = text_size[1] + 20
        bg_rect_top_left = (width - bg_rect_width - 10, 10)
        bg_rect_bottom_right = (width - 10, bg_rect_height + 10)

        # Draw semi-transparent background
        overlay = drawn_frame.copy()
        cv2.rectangle(overlay, bg_rect_top_left, bg_rect_bottom_right, (0, 0, 0), -1)
        t = 0.25
        cv2.addWeighted(overlay, t, drawn_frame, 1-t, 0, drawn_frame)

        # Draw text with glass effect
        text_pos = (width - text_size[0] - 20, bg_rect_height - 5)
        cv2.putText(drawn_frame, fps_text, text_pos, font, font_scale, (200, 200, 200), font_thickness + 1, cv2.LINE_AA)
        cv2.putText(drawn_frame, fps_text, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return drawn_frame

    def update_fps(self, frame: np.ndarray):
        if self.prev_frame_time:
            self.fps = 1 / (self.new_frame_time - self.prev_frame_time)
        else:
            self.fps = 0

        self.last_fps_update = time.time()  # Update timestamp

    def annotate_with_yolo(self, frame):
        results = self.yolo_vision_model.model.predict(frame, conf=0.7, verbose=False)

        annotator = Annotator(frame)
        object_positions = {}

        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                center_x = int((b[0] + b[2]) // 2)
                center_y = int((b[1] + b[3]) // 2)

                c = box.cls
                object_label = self.yolo_vision_model.model.names[int(c)]
                prediction_confidence = box.conf.item()
                annotator.box_label(b, f"{object_label} {prediction_confidence:.2f}", color=(83, 130, 46))
                object_positions[object_label] = b.tolist()

        processed_frame = annotator.result()

        return processed_frame, object_positions

    def annotate_objects_to_frame(self, frame):
        if not self.object_positions:
            return frame

        annotator = Annotator(frame)

        for object_label, object_positions in self.object_positions.items():
            annotator.box_label(object_positions, f"{object_label}", color=(83, 130, 46))

        processed_frame = annotator.result()

        return processed_frame

    def detect_aruco_markers(self, frame):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)

        # Initialize the dictionary to store marker ID and bounding box coordinates
        markers = {}

        if ids is not None:
            for i, corner in enumerate(corners):
                # Get the coordinates of the bounding box
                x_min = int(min(corner[0][:, 0]))
                y_min = int(min(corner[0][:, 1]))
                x_max = int(max(corner[0][:, 0]))
                y_max = int(max(corner[0][:, 1]))

                # Add to dictionary
                markers[f"aruco-marker-{int(ids[i][0])}"] = (x_min, y_min, x_max, y_max)

        if markers:
            for marker_id, positions in markers.items():
                if marker_id not in self.object_positions:
                    self.object_positions[marker_id] = positions
            self.aruco_last_detected_at = time.time()
        elif (time.time() - self.aruco_last_detected_at) >= 2:
            for k in self.object_positions.keys():
                if k.startswith("aruco-marker-"):
                    del self.object_positions[k]

        return markers

    def run(self):
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

        aruco_parameters = cv2.aruco.DetectorParameters()
        aruco_parameters.minDistanceToBorder = 0
        aruco_parameters.adaptiveThreshWinSizeMax = 400

        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)

        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            logging.error("Failed to open camera")
            return

        while self._run_flag:
            ret, frame = cap.read()

            if not ret:
                time.sleep(1/100)
                continue

            self.new_frame_time = time.time()

            # Convert to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Update FPS every 3 seconds
            if self.last_fps_update is None or (time.time() - self.last_fps_update) >= self.frame_update_interval:
                self.update_fps(frame)

            self.prev_frame_time = self.new_frame_time

            # processed_frame, object_positions = self.annotate_with_yolo(frame.copy())
            processed_frame = frame.copy()

            # Pass 1: No op, aruco
            self.detect_aruco_markers(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2GRAY))

            # Pass 2: Frame contains all objects
            processed_frame = self.annotate_objects_to_frame(processed_frame)

            # Pass 3: Frame contains fps info
            processed_frame = self.draw_fps(processed_frame, self.fps)

            # self.update_object_positions_signal.emit(object_positions)

            self.change_pixmap_signal.emit((frame, processed_frame))

            time.sleep(1/500)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class LogStreamThread(QThread):
    update_log_signal = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.log_queue = kwargs['log_queue']
        self._run_flag = True

    def run(self):
        while self._run_flag:
            try:
                text = self.log_queue.get(timeout=1)
                self.update_log_signal.emit(text)
            except queue.Empty:
                pass

    def stop(self):
        self._run_flag = False
        self.wait()


class ObjectPositionStreamThread(QThread):
    update_signal = pyqtSignal(bool)

    def __init__(self, **kwargs):
        super().__init__()
        self.object_positions = kwargs['object_positions']
        self._run_flag = True

    def run(self):
        while self._run_flag:
            self.update_signal.emit(True)
            time.sleep(1/100)

    def stop(self):
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self, **kwargs):
        super().__init__()
        self.setWindowTitle("Dobot Feed")
        self.display_width = 640
        self.display_height = 480

        # Image label
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        # Upper text box for logs
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setFixedWidth(480)
        log_height = math.floor(self.display_height * 0.7)
        self.log_text.setFixedHeight(log_height)  # Half of display_height

        # Lower text box for dictionary display
        self.object_position_text = QTextEdit(self)
        self.object_position_text.setReadOnly(True)
        self.object_position_text.setFixedWidth(480)
        self.object_position_text.setFixedHeight(self.display_height - log_height)  # Half of display_height
        self.object_positions = kwargs['object_positions']
        self.object_position_updated_at = time.time()

        # Layout setup
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.log_text)
        right_layout.addWidget(self.object_position_text)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        # Disable window close button and resize handles
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        # self.setFixedSize(self.sizeHint())

        self.latest_frame = kwargs['latest_frame']

        self.camera_thread = CameraThread(**kwargs)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.update_object_positions_signal.connect(self.update_object_positions)
        self.camera_thread.start()

        self.log_stream_thread = LogStreamThread(**kwargs)
        self.log_stream_thread.update_log_signal.connect(self.update_log_text)
        self.log_stream_thread.start()

        self.object_position_stream_thread = ObjectPositionStreamThread(**kwargs)
        self.object_position_stream_thread.update_signal.connect(self.update_object_positions_n)
        self.object_position_stream_thread.start()

        logging.debug("App initialized")

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.log_stream_thread.stop()
        self.object_position_stream_thread.stop()
        event.accept()

    @pyqtSlot(bool)
    def update_object_positions_n(self, _):
        now = time.time()

        # update positions every 2 seconds
        if (now - self.object_position_updated_at) >= 2:
            self.object_position_updated_at = now

            self.object_position_text.clear()

            for key, value in self.object_positions.items():
                self.object_position_text.insertHtml(
                    f'<span style="color:black;">{key}:</span> '
                    f'<span style="color:blue;">x_min={value[0]:.3f}</span> '
                    f'<span style="color:blue;">y_min={value[1]:.3f}</span> '
                    f'<span style="color:purple;">x_max={value[2]:.3f}</span> '
                    f'<span style="color:purple;">y_max={value[3]:.3f}</span>'
                )
                self.object_position_text.insertHtml("<br/>")

    @pyqtSlot(dict)
    def update_object_positions(self, new_object_positions: dict):
        self.object_positions.clear()
        self.object_positions.update(new_object_positions)

        now = time.time()

        # update positions every 2 seconds
        if (now - self.object_position_updated_at) >= 2:
            self.object_position_updated_at = now

            self.object_position_text.clear()

            for key, value in new_object_positions.items():
                self.object_position_text.insertHtml(
                    f'<span style="color:black;">{key}:</span> '
                    f'<span style="color:blue;">x_min={value[0]:.3f}</span> '
                    f'<span style="color:blue;">y_min={value[1]:.3f}</span> '
                    f'<span style="color:purple;">x_max={value[2]:.3f}</span> '
                    f'<span style="color:purple;">y_max={value[3]:.3f}</span>'
                )
                self.object_position_text.insertHtml("<br/>")

    @pyqtSlot(tuple)
    def update_image(self, frames):
        try:
            original_frame, processed_frame = frames
            self.latest_frame.update_image(original_frame)

            self.image_label.setPixmap(self.convert_cv_qt(processed_frame))
        except Exception as e:
            logging.error(f"Error updating image: {str(e)}")

    @pyqtSlot(str)
    def update_log_text(self, text):
        self.log_text.insertHtml(text)
        self.log_text.insertHtml("<br>")  # Add a line break after each message
        self.log_text.moveCursor(QTextCursor.End)
        logging.debug(f"Text updated: {text}")

    def convert_cv_qt(self, frame):
        # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def process_transcription(transcription, log_queue):
    if not transcription:
        return

    transcription_len = len(transcription)

    if transcription_len < 10 and _exit_pattern.search(transcription):
        print_system_message("Exiting...")
        sys.exit(0)

    if transcription_len < 10:
        log_to_queue(log_queue, "Message to small to be processed", color=Fore.YELLOW, log_level=logging.WARNING)
        return

    log_to_queue(log_queue, transcription, color=Fore.GREEN, log_level=logging.INFO, scope="user")

    full_content = ""

    for out in lm.forward([
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": transcription},
    ]):
        if out:
            full_content += out["content"]
            # print("Got token", out["content"])

    # try some patches if necessary.
    if full_content.startswith("```json"):
        full_content = full_content.removeprefix("```json").removesuffix("```")

    try:
        tool_calls = json.loads(full_content)
        return tool_calls
    except ValueError:
        log_to_queue(log_queue, f"Invalid model output: {full_content}", color=Fore.RED, log_level=logging.ERROR)

    return None


# Thread-safe flag for graceful shutdown
shutdown_flag = Event()


def producer(**kwargs):
    action_queue = kwargs['action_queue']
    log_queue = kwargs['log_queue']
    robot_working = kwargs['robot_working']
    speech_recognizer = SpeechRecognizer()

    def _is_paused():
        return robot_working.is_set()

    for transcription in speech_recognizer.run(is_paused=_is_paused):
        actions = process_transcription(transcription, log_queue=log_queue)

        if actions:
            robot_working.set()
            action_queue.put_nowait(actions)

        if shutdown_flag.is_set():
            break


def process_actions(actions: List[dict]) -> None:
    ...


def consumer(**kwargs):
    action_queue = kwargs['action_queue']
    florence_vision_model = MicrosoftFlorence2()
    latest_frame = kwargs['latest_frame']
    log_queue = kwargs['log_queue']
    object_positions = kwargs['object_positions']
    robot_working = kwargs['robot_working']

    while not shutdown_flag.is_set():
        try:
            # print(object_positions)
            actions = action_queue.get(timeout=1)
            log_to_queue(log_queue, f"Consumed: {actions}")

            # frame = Image.fromarray(latest_frame.get_image())
            # _, result = florence_vision_model.run_inference(frame, item)
            # detections = florence_vision_model.inference_to_sv_detections(result, frame)
            #
            # new_object_positions = {}
            #
            # for detection in detections:
            #     new_object_positions[detections[-1]['class_name'][0]] = detection[0].tolist()
            #
            # object_positions.clear()
            # object_positions.update(new_object_positions)
            #
            # robot_working.clear()
        except queue.Empty:
            pass


def camera_feed_handler(**kwargs):
    app = QApplication(sys.argv)
    window = App(**kwargs)
    window.show()

    return app.exec_()


def signal_handler(signum, frame):
    if not shutdown_flag.is_set():
        shutdown_flag.set()
        print("\nCtrl+C detected. Initiating shutdown...")


def _real_main(**kwargs):
    signal.signal(signal.SIGINT, signal_handler)

    manager = Manager()

    # Create queues for communication
    action_queue = Queue()
    log_queue = Queue()
    object_positions = manager.dict()
    robot_working = Event()

    process_kwargs = {
        'action_queue': action_queue,
        'latest_frame': ImageSharer(),
        'log_queue': log_queue,
        'object_positions': object_positions,
        'robot_working': robot_working,
    }

    # Create producer and consumer tasks
    producer_process = Process(target=producer, kwargs=process_kwargs)
    consumer_process = Process(target=consumer, kwargs=process_kwargs)
    camera_feed_process = Process(target=camera_feed_handler, kwargs=process_kwargs)

    producer_process.start()
    consumer_process.start()
    camera_feed_process.start()

    if not shutdown_flag.is_set():
        time.sleep(0.1)

    # Wait for processes to finish
    producer_process.join()
    consumer_process.join()
    camera_feed_process.join()

    print("Shutdown complete")


@click.command()
def main(**kwargs):
    _real_main(**kwargs)
