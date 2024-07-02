import numpy as np


def landmark_to_numpy(landmark):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark]).flatten()


def get_bounding_box(frame_dim, hand_landmarks):
    w, h = frame_dim
    x_values = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_values = [landmark.y * h for landmark in hand_landmarks.landmark]
    bbox = int(min(x_values)), int(min(y_values)), int(max(x_values)), int(max(y_values))

    return bbox


def landmark_to_ratio(landmark, bounding_rect):
    x_min, y_min, x_max, y_max = bounding_rect
    landmark_x, landmark_y = landmark
    ratio_x = (landmark_x - x_min) / (x_max - x_min)
    ratio_y = (landmark_y - y_min) / (y_max - y_min)
    return ratio_x, ratio_y
