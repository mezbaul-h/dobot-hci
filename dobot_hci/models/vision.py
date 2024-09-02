import os
import warnings
from typing import Union
from unittest.mock import patch

import numpy as np

from PIL import Image
import supervision as sv
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

            from transformers import AutoProcessor, AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
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
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task, image_size=image.size)

        return generated_text, parsed_answer

    @staticmethod
    def inference_to_sv_detections(inference, frame: Image.Image) -> sv.Detections:
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=inference,
            resolution_wh=frame.size
        )

        return detections


class MetaSAM2:
    def __init__(self):
        self.device = settings.TORCH_DEVICE
        self.config_dir = settings.CUSTOM_MODELS_DIR / "sam2"
        self.checkpoint_dir = settings.CUSTOM_MODELS_DIR / "sam2"

        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.config = self.config_dir / "sam2_hiera_s.yaml"
        self.checkpoint = self.checkpoint_dir / "sam2_hiera_small.pt"
        self.image_model = SAM2ImagePredictor(sam_model=build_sam2(self.config, self.checkpoint, device=self.device))
        self.video_model = build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)

    def run_inference(
        self,
        image: Image.Image,
        detections: sv.Detections
    ) -> sv.Detections:
        image = np.array(image.convert("RGB"))

        self.image_model.set_image(image)

        mask, score, _ = self.image_model.predict(box=detections.xyxy, multimask_output=False)

        # dirty fix; remove this later
        if len(mask.shape) == 4:
            mask = np.squeeze(mask)

        detections.mask = mask.astype(bool)

        return detections


class YOLO:
    def __init__(self, model_name: str = "yolov8s-worldv2.pt"):
        from ultralytics import YOLO as UL_YOLO

        self.model = UL_YOLO(model_name, verbose=False)

    def get_classes(self):
        return self.model.names
