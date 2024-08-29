# from .recorders import ObjectRecorder
import json
import logging
import re
import sys

import click
from colorama import Fore

from .models.language import OllamaLM
from .models.speech import SpeechRecognizer
from .utils import print_system_message

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


def process_transcription(transcription):
    if not transcription:
        return

    transcription_len = len(transcription)

    if transcription_len < 10 and _exit_pattern.search(transcription):
        print_system_message("Exiting...")
        sys.exit(0)

    if transcription_len < 10:
        print_system_message("Message to small to be processed", color=Fore.YELLOW, log_level=logging.WARNING)
        return

    print_system_message(transcription, color=Fore.GREEN, log_level=logging.INFO, scope="user")

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
        print(tool_calls)
    except ValueError:
        print_system_message(f"Invalid model output: {full_content}", color=Fore.RED, log_level=logging.ERROR)


@click.command()
def main(**kwargs):
    print(kwargs)

    sr = SpeechRecognizer()

    for transcription in sr.run():
        process_transcription(transcription)

    # with ObjectRecorder() as recorder:
    #     recorder.run()
