import warnings

import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class Florence2FT:
    def __init__(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

    def infer(self, image: Image):
        prompt = "<MORE_DETAILED_CAPTION>"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

        return parsed_answer
