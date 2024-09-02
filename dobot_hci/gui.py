import logging
import math
import queue
import time
from multiprocessing import Event
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget
from ultralytics.utils.plotting import Annotator


class QThreadBase(QThread):
    def __init__(self):
        super().__init__()
        self._run_flag = True

    def stop(self):
        self._run_flag = False
        self.wait()


class CameraThread(QThreadBase):
    change_pixmap_signal = pyqtSignal(tuple)
    update_object_positions_signal = pyqtSignal(dict)

    def __init__(self, **kwargs):
        super().__init__()

        self.shutdown_event: Optional[Event] = kwargs.get("shutdown_event")
        self.camera_index = kwargs.get("camera_index", 0)
        self.object_positions = kwargs["object_positions"]
        self.fps = None
        self.new_frame_time = None
        self.prev_frame_time = None
        self.last_fps_update = None
        self.frame_update_interval = 2  # seconds
        self.aruco_detector = None
        self.aruco_last_detected_at = time.time()

        # CV
        self.capture = None

        # RS
        self.realsense_pipeline = None
        self.use_realsense = kwargs.get("use_realsense", False)

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
        cv2.addWeighted(overlay, t, drawn_frame, 1 - t, 0, drawn_frame)

        # Draw text with glass effect
        text_pos = (width - text_size[0] - 20, bg_rect_height - 5)
        cv2.putText(
            drawn_frame, fps_text, text_pos, font, font_scale, (200, 200, 200), font_thickness + 1, cv2.LINE_AA
        )
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

    def _init_camera_feed(self):
        if self.use_realsense:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()

            config = rs.config()

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

            # Start streaming
            self.pipeline.start(config)
        else:
            self.capture = cv2.VideoCapture(self.camera_index)

            if not self.capture.isOpened():
                raise Exception("Failed to open camera")

    def _destroy_camera_feed(self):
        if self.use_realsense:
            self.pipeline.stop()
        else:
            self.capture.release()

    def get_frame(self):
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame:
                return False, None

            # Convert images to numpy arrays
            color_frame = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())

            return True, color_frame
        else:
            ret, frame = self.capture.read()

            # Convert to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return ret, frame

    def run(self):
        self._init_camera_feed()

        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

        aruco_parameters = cv2.aruco.DetectorParameters()
        aruco_parameters.minDistanceToBorder = 0
        aruco_parameters.adaptiveThreshWinSizeMax = 400

        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)

        while self._run_flag:
            ret, frame = self.get_frame()

            if not ret:
                time.sleep(1 / 100)
                continue

            self.new_frame_time = time.time()

            if not self.use_realsense:
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

            time.sleep(1 / 800)

        self._destroy_camera_feed()


class LogStreamThread(QThreadBase):
    update_log_signal = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()

        self.log_queue = kwargs["log_queue"]

    def run(self):
        while self._run_flag:
            try:
                text = self.log_queue.get(timeout=1)
                self.update_log_signal.emit(text)
            except queue.Empty:
                pass


class ObjectPositionStreamThread(QThreadBase):
    update_signal = pyqtSignal(bool)

    def __init__(self, **kwargs):
        super().__init__()

        self.object_positions = kwargs["object_positions"]

    def run(self):
        while self._run_flag:
            self.update_signal.emit(True)
            time.sleep(1 / 100)


class GUIApplication(QWidget):
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
        self.log_text.setStyleSheet("background-color: #E2E2E2;")

        # Lower text box for dictionary display
        self.object_position_text = QTextEdit(self)
        self.object_position_text.setReadOnly(True)
        self.object_position_text.setFixedWidth(480)
        self.object_position_text.setFixedHeight(self.display_height - log_height)  # Half of display_height
        self.object_position_text.setStyleSheet("background-color: #E2E2E2;")
        self.object_positions = kwargs["object_positions"]
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

        self.latest_frame = kwargs["latest_frame"]

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
