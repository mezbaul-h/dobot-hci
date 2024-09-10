import logging
import sys
import time
from typing import Optional

import torch.multiprocessing as mp
from colorama import Fore
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

from ..audio import AudioIO
from ..settings import settings
from ..utils import ProcessTime, log_to_queue


class SpeechRecognizer:
    def __init__(self, **kwargs) -> None:
        self.device = "cpu"
        self.log_queue: Optional[mp.Queue] = kwargs.get("log_queue")
        self.shutdown_flag: Optional[mp.Event] = kwargs.get("shutdown_flag")
        self.command_classifier = pipeline(
            "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=self.device
        )
        self.audio_io = AudioIO(
            log_queue=self.log_queue,
            shutdown_flag=self.shutdown_flag,
        )
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            chunk_length_s=10,
            device=self.device,
            model="openai/whisper-small.en",
            # token=settings.HF_TOKEN,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def transcribe(self):
        sampling_rate = self.transcriber.feature_extractor.sampling_rate

        audio_data = self.audio_io.record_audio()

        if audio_data:
            with ProcessTime("Transcription"):
                transcription = self.transcriber(audio_data, batch_size=8)

            if "text" in transcription:
                return transcription["text"].strip()

        return None

    def wait_on_wake_command(
        self,
        wake_word="marvin",
        prob_threshold=0.5,
        chunk_length_s=2.0,
        stream_chunk_s=0.25,
        debug=False,
    ):
        if wake_word not in self.command_classifier.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the "
                f"set {self.command_classifier.model.config.label2id.keys()}."
            )

        sampling_rate = self.command_classifier.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        log_to_queue(self.log_queue, "Listening for wake word...", color=Fore.BLUE, log_level=logging.ERROR)

        for prediction in self.command_classifier(mic):
            prediction = prediction[0]

            if debug:
                print(prediction)

            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threshold:
                    return True

            if self.shutdown_flag and self.shutdown_flag.is_set():
                break

    def run(self, is_paused=None):
        while True:
            if is_paused and is_paused():
                time.sleep(1 / 100)
                continue

            if self.shutdown_flag and self.shutdown_flag.is_set():
                break

            self.wait_on_wake_command(debug=True)

            if self.shutdown_flag and self.shutdown_flag.is_set():
                break

            transcription = self.transcribe()

            yield transcription
