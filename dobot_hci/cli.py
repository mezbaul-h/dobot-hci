import json
import logging
import queue
import re
import signal
import sys
import time
import traceback
from typing import List

import click
import numpy as np
import torch.multiprocessing as mp
from colorama import Fore
from PIL import Image
from PyQt5.QtWidgets import QApplication

from .gui import GUIApplication
from .models.language import OllamaLM
from .models.speech import SpeechRecognizer
from .models.vision import MicrosoftFlorence2
from .robot_controls import execute_robot_action
from .utils import log_to_queue, print_system_message

_exit_pattern = re.compile(r"\b(exit|quit|stop)\b", re.IGNORECASE)

_llm_model = OllamaLM()
_system_prompt = """
You are an AI assistant integrated with a robot. Your role is to interpret user instructions and translate them into a series of actionable commands for the robot. The robot can perform the following actions:

1. move_object_near(source, target)
2. move_object_on_top_of(source, target)
3. move_object_to_down(object_name)
4. move_object_to_down_of(source_object_name, target_object_name)
5. move_object_to_left(object_name)
6. move_object_to_left_of(source_object_name, target_object_name)
7. move_object_to_right(object_name)
8. move_object_to_right_of(source_object_name, target_object_name)
9. move_object_to_up(object_name)
10. move_object_to_up_of(source_object_name, target_object_name)

Your task is to:
1. Interpret the user's instruction.
2. Translate the instruction into a series of robot actions using the available commands.
3. Return ONLY a JSON array containing the sequence of actions required to fulfill the instruction.
4. If the instruction cannot be translated into available actions, return a JSON object with a single key "error" and a value explaining why the instruction cannot be executed.

Do not include any explanation, conversation, or additional text in your response. Your output should be valid JSON and nothing else.

Examples:
User: "Move the apple to the left of the orange"
Response: ["move_object_to_left_of(apple, orange)"]

User: "Put the book on the shelf"
Response: ["move_object_on_top_of(book, shelf)"]

User: "Grab me the book and then move the orange to the left of the apple"
Response: ["move_object_on_top_of(book, hand)", "move_object_to_left_of(orange, apple)"]

User: "Make the robot dance"
Response: {"error": "The robot does not have a 'dance' function. Available actions are limited to moving objects."}

User: "How are you"
Response: {"error": "I cannot fulfill that request."}

User: "Write a poem"
Response: {"error": "I cannot fulfill that request."}
"""


class ImageSharer:
    def __init__(self, max_size=(1920, 1080, 3)):  # Adjust max_size as needed
        self.max_size = max_size
        self.shared_array = mp.Array("B", max_size[0] * max_size[1] * max_size[2])
        self.metadata = mp.Array("i", 3)  # To store current image dimensions

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
        shared_np_array[: h * w * c] = cv_img.ravel()

    def get_image(self):
        h, w, c = self.metadata
        shared_np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8)
        return shared_np_array[: h * w * c].reshape(h, w, c)


def process_transcription(transcription, log_queue, shutdown_flag):
    if not transcription:
        return None

    transcription_len = len(transcription)

    # if transcription_len < 10 and _exit_pattern.search(transcription):
    #    log_to_queue(log_queue, "Exit pattern detected. Exiting...", color=Fore.BLUE, log_level=logging.INFO)
    #    _set_shutdown_flag()
    #    return None

    if transcription_len < 10:
        log_to_queue(log_queue, "Message to small to be processed", color=Fore.YELLOW, log_level=logging.WARNING)
        return None

    log_to_queue(log_queue, transcription, color=Fore.GREEN, log_level=logging.INFO, scope="user")

    full_content = ""

    for out in _llm_model.forward(
        [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": transcription},
        ]
    ):
        if out:
            full_content += out["content"]

        if shutdown_flag.is_set():
            return None

    # try some patches if necessary.
    if full_content.startswith("```json"):
        full_content = full_content.removeprefix("```json").removesuffix("```")

    try:
        tool_calls = json.loads(full_content)

        return tool_calls
    except ValueError:
        log_to_queue(log_queue, f"Invalid model output: {full_content}", color=Fore.RED, log_level=logging.ERROR)

    return None


def transcription_handler(**kwargs):
    """
    Handles voice input and generates list of actions.
    """
    action_queue = kwargs["action_queue"]
    log_queue = kwargs["log_queue"]
    robot_working_flag = kwargs["robot_working_flag"]
    shutdown_flag = kwargs["shutdown_flag"]
    speech_recognizer = SpeechRecognizer(
        log_queue=log_queue,
        shutdown_flag=shutdown_flag,
    )

    def _is_paused():
        return robot_working_flag.is_set()

    for transcription in speech_recognizer.run(is_paused=_is_paused):
        actions = process_transcription(transcription, log_queue=log_queue, shutdown_flag=shutdown_flag)

        if actions:
            # time.sleep(10)
            robot_working_flag.set()
            action_queue.put_nowait(actions)

        if shutdown_flag.is_set():
            break

    # while True:
    #     if _is_paused():
    #         continue
    #
    #     robot_working_flag.set()
    #     action_queue.put_nowait(["move_object_on_top_of(watermelon, hand)"])
    #     # action_queue.put_nowait(["move_object_to_up(scissors)"])
    #
    #     if shutdown_flag.is_set():
    #         break
    #
    #     time.sleep(1)

    speech_recognizer.audio_io.close()


def parse_actions(actions: List[dict]):
    if isinstance(actions, dict):
        actions = [actions]

    parsed_actions = []

    if "error" in actions[0]:
        return None

    for action in actions:
        if isinstance(action, dict):
            action = action["action"]

        arguments = [item.strip() for item in action.split("(")[-1][:-1].split(",") if item]
        method = action.split("(")[0]

        parsed_actions.append(
            {
                "arguments": arguments,
                "method": method,
            }
        )

    return parsed_actions


def robot_controller(**kwargs):
    """
    1. Consumes actions
    2. Uses florence's open vocab detection to annotate target objects
    3. Makes robot to perform actions
    """
    action_queue = kwargs["action_queue"]
    florence_vision_model = MicrosoftFlorence2()
    latest_frame = kwargs["latest_frame"]
    log_queue = kwargs["log_queue"]
    object_positions = kwargs["object_positions"]
    robot_working_flag = kwargs["robot_working_flag"]
    shutdown_flag = kwargs["shutdown_flag"]

    while not shutdown_flag.is_set():
        try:
            # print(object_positions)
            actions = action_queue.get(timeout=1)
            log_to_queue(log_queue, f"Consumed: {actions}")

            parsed_actions = parse_actions(actions)

            if not parsed_actions:
                robot_working_flag.clear()
                continue

            classification_objects = []

            for parsed_action in parsed_actions:
                classification_objects.extend(parsed_action["arguments"])

            # Position with florence
            frame = Image.fromarray(latest_frame.get_image())
            new_object_positions = {}

            for classification_object in classification_objects:
                _, result = florence_vision_model.run_inference(frame, classification_object)
                detections = florence_vision_model.inference_to_sv_detections(result, frame)

                for detection in detections:
                    new_object_positions[detections[-1]["class_name"][0]] = detection[0].tolist()

            object_positions.update(new_object_positions)

            for parsed_action in parsed_actions:
                arguments = parsed_action["arguments"]
                method = parsed_action["method"]

                log_to_queue(log_queue, f"Executing {method} with {arguments}")
                execute_robot_action(method, arguments, object_positions, log_queue)
                log_to_queue(log_queue, f"Done executing {method}")
        except queue.Empty:
            pass
        except (AttributeError, ValueError) as exc:
            print(traceback.format_exc())
            log_to_queue(log_queue, f"Failed execution; reason: {exc}", color=Fore.RED, log_level=logging.ERROR)
        finally:
            robot_working_flag.clear()

            # Do not remove end-effector position
            for k in object_positions.keys():
                if not k.startswith("aruco-marker-"):
                    del object_positions[k]


def ui_handler(**kwargs):
    app = QApplication([])
    window = GUIApplication(**kwargs)

    window.show()

    _return_code = app.exec_()

    # GUI exited, mark shutdown flag for other processes.
    kwargs["shutdown_flag"].set()

    return _return_code


def signal_handler(signum, frame):
    pass


def _real_main(**kwargs):
    signal.signal(signal.SIGINT, signal_handler)

    manager = mp.Manager()

    process_kwargs = {
        "action_queue": mp.Queue(),
        "latest_frame": ImageSharer(),
        "log_queue": mp.Queue(),
        "object_positions": manager.dict(),
        "robot_working_flag": mp.Event(),
        "shutdown_flag": mp.Event(),
        "use_realsense": kwargs["use_realsense"],
    }

    processes = [
        mp.Process(target=robot_controller, kwargs=process_kwargs),
        mp.Process(target=transcription_handler, kwargs=process_kwargs),
        mp.Process(target=ui_handler, kwargs=process_kwargs),
    ]

    # Start processes
    for process in processes:
        process.start()

    if not process_kwargs["shutdown_flag"].is_set():
        time.sleep(1 / 10)

    # Wait for processes to finish
    for process in processes:
        process.join()

    print("Shutdown complete")


@click.command()
@click.option(
    "-rs", "--use-realsense", is_flag=True, show_default=True, default=False, help="Use Intel RealSense Camera."
)
def main(**kwargs):
    _real_main(**kwargs)
