from pathlib import Path

BODY_POSE_LANDMARK_COUNT = 33

HAND_GESTURE_LANDMARK_COUNT = 21

BODY_POSE_LABELS = [
    "thinking",
    "victory",
]

HAND_GESTURE_LABELS = [
    "beckoning",  # come here
    "closed_fist",
    "handshake",
    "open_palm",
    # "point",
    "thumbs_down",
    "thumbs_up",
]

LABELS = BODY_POSE_LABELS + HAND_GESTURE_LABELS

PACKAGE_ROOT_DIR = Path(__file__).parent

YOLO_ROOT_DIR = PACKAGE_ROOT_DIR.parent / "yolo"

TRAINING_DATA_DIR = PACKAGE_ROOT_DIR / "training_data"
