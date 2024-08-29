import logging
import sys

from transformers import pipeline
import torch
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from ..audio import AudioIO
from ..utils import print_system_message

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SpeechRecognizer:
    def __init__(self, **kwargs) -> None:
        self.command_classifier = pipeline(
            "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
        )
        self.audio_io = AudioIO()
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            chunk_length_s=10,
            device=device,
            model="openai/whisper-small.en",
            # token=settings.HF_TOKEN,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def transcribe(self):
        sampling_rate = self.transcriber.feature_extractor.sampling_rate

        audio_data = self.audio_io.record_audio()

        if audio_data:
            transcription = self.transcriber(audio_data, batch_size=8)

            if 'text' in transcription:
                return transcription['text'].strip()

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
                f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {self.command_classifier.model.config.label2id.keys()}."
            )

        sampling_rate = self.command_classifier.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        print_system_message("Listening for wake word...", log_level=logging.INFO)

        for prediction in self.command_classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threshold:
                    return True

    def run(self):
        while True:
            self.wait_on_wake_command(debug=False)
            transcription = self.transcribe()

            yield transcription
