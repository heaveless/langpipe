import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class LangPipe:
    def _load_model_and_processor(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def _load_dataset(self):
        self.dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    def _load_audio(self):
        audio_input, sample_rate = sf.read(self.dataset[0]["file"])
        self.input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    def _retrieve_logits_and_take_argmax(self):
        logits = self.model(self.input_values).logits
        self.predicted_ids = torch.argmax(logits, dim=-1)

    def _transcribe(self):
        self.transcription = self.processor.decode(self.predicted_ids[0])

    def _fine_tine(self):
        target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

        with self.processor.as_target_processor():
            labels = self.processor(target_transcription, return_tensors="pt").input_ids

        loss = self.model(self.input_values, labels=labels).loss
        loss.backward()

    def run(self):
        self._load_model_and_processor()
        self._load_dataset()
        self._load_audio()
        self._retrieve_logits_and_take_argmax()
        self._transcribe()
        self._fine_tine()