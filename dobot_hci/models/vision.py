import os
import warnings
from typing import Union
from unittest.mock import patch

import numpy as np
import supervision as sv
from PIL import Image
from transformers.dynamic_module_utils import get_imports

from dobot_hci.settings import settings


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """
    Work around for: https://huggingface.co/microsoft/phi-1_5/discussions/72
    """
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)

    imports = get_imports(filename)
    imports.remove("flash_attn")

    return imports


class MicrosoftFlorence2:
    def __init__(self, model_name: str = "microsoft/Florence-2-base") -> None:
        self.device = settings.TORCH_DEVICE

        with warnings.catch_warnings(), patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            warnings.simplefilter("ignore")

            from transformers import AutoModelForCausalLM, AutoProcessor

            self.model = (
                AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
            )
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def run_inference(self, image: Image.Image, open_vocabulary_prompt: str = None):
        task = "<OPEN_VOCABULARY_DETECTION>"
        prompt = f"{task}{open_vocabulary_prompt or ''}".strip()

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            # do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task, image_size=image.size)

        return generated_text, parsed_answer

    @staticmethod
    def inference_to_sv_detections(inference, frame: Image.Image) -> sv.Detections:
        detections = sv.Detections.from_lmm(lmm=sv.LMM.FLORENCE_2, result=inference, resolution_wh=frame.size)

        return detections


class YOLO:
    def __init__(self, model_name: str = "yolov8s-worldv2.pt"):
        from ultralytics import YOLO as UL_YOLO

        self.model = UL_YOLO(model_name, verbose=False)

    def get_classes(self):
        return self.model.names
