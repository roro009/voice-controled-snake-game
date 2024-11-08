# voice_control.py
import sounddevice as sd
import numpy as np
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from multiprocessing import Value
import ctypes
import time
import nemo
import wave
import os
import tempfile
from pytorch_lightning.core.memory import ModelSummary
from torchmetrics.utilities.data import get_num_classes
from torchmetrics.classification import F1Score
from collections import Callable

# Load the NeMo ASR Model from a local .nemo file to avoid Hugging Face dependencies
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_large")

# Parameters
SAMPLE_RATE = 16000  # Required sample rate for NeMo ASR models
DURATION = 1  # Duration of each audio chunk in seconds

def recognize_direction(snake_direction):
    print("Starting real-time voice recognition...")

    while True:
        print("Listening for command...")
        
        # Record audio for DURATION seconds using sounddevice
        audio_input = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((audio_input * 32767).astype(np.int16).tobytes())

        try:
            # Perform ASR transcription by passing the temporary file path
            transcription = asr_model.transcribe([temp_filename])[0].lower()
            print(f"Recognized command: {transcription}")
            
            # Set snake direction based on the recognized command
            if "up" in transcription and snake_direction.value != 2:  # Prevent reverse direction
                snake_direction.value = 0
            elif "down" in transcription and snake_direction.value != 0:
                snake_direction.value = 2
            elif "left" in transcription and snake_direction.value != 1:
                snake_direction.value = 3
            elif "right" in transcription and snake_direction.value != 3:
                snake_direction.value = 1
        except Exception as e:
            print(f"Error in recognition: {e}")
        finally:
            # Clean up the temporary file
            os.remove(temp_filename)

        time.sleep(0.1)  # Small delay to prevent high CPU usage

if __name__ == "__main__":
    snake_direction = Value(ctypes.c_int, 1)  # Initial direction is "RIGHT"
    recognize_direction(snake_direction)