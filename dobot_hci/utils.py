"""
This module provides utility classes and functions.
"""

import logging
import re
from multiprocessing import Queue

from colorama import Fore, Style

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def print_system_message(
    message: str, color: str = Fore.BLUE, log_level: int = logging.DEBUG, scope: str = "system"
) -> None:
    """
    Print a message with a colored system prompt.

    Args:
        message: The message to be printed.
        color: The color code for the message text (e.g., Fore.BLUE).
            Defaults to Fore.BLUE.
        log_level: The logging level for the message (e.g., logging.DEBUG).
            Defaults to logging.DEBUG.
    """
    logger.log(log_level, f"{Style.BRIGHT}{Fore.YELLOW}[{scope}]> {Style.NORMAL}{color}{message}{Style.RESET_ALL}")


def ansi_to_html(ansi_code):
    """Converts an ANSI color code to an HTML RGB color string."""
    # Match the ANSI code
    match = re.match(r"(\d+)", str(ansi_code))

    if not match:
        return None

    # Determine the RGB values based on the ANSI code
    code = int(match.group(1))
    if 30 <= code <= 37:
        # Regular colors
        r, g, b = [0, 0, 0]
        if code == 30:
            r, g, b = 0, 0, 0
        elif code == 31:
            r, g, b = 255, 0, 0
        elif code == 32:
            r, g, b = 0, 255, 0
        elif code == 33:
            r, g, b = 255, 255, 0
        elif code == 34:
            r, g, b = 0, 0, 255
        elif code == 35:
            r, g, b = 255, 0, 255
        elif code == 36:
            r, g, b = 0, 255, 255
        elif code == 37:
            r, g, b = 255, 255, 255
    elif 90 <= code <= 97:
        # Bright colors
        r, g, b = [128, 128, 128]
        if code == 90:
            r, g, b = 128, 128, 128
        elif code == 91:
            r, g, b = 255, 128, 128
        elif code == 92:
            r, g, b = 128, 255, 128
        elif code == 93:
            r, g, b = 255, 255, 128
        elif code == 94:
            r, g, b = 128, 128, 255
        elif code == 95:
            r, g, b = 255, 128, 255
        elif code == 96:
            r, g, b = 128, 255, 255
        elif code == 97:
            r, g, b = 255, 255, 255
    else:
        return None

    # Create the HTML RGB color string
    return f"rgb({r}, {g}, {b})"


def log_to_queue(
    queue: Queue, message: str, color: str = Fore.BLUE, log_level: int = logging.DEBUG, scope: str = "system"
) -> None:
    color = ansi_to_html(color.split("[")[-1])

    queue.put_nowait(f'<span style="color:{color};"><b>[{scope}]</b>&gt; {message}</span>')
