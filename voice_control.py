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
from difflib import SequenceMatcher

# Load the NeMo ASR Model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_large")

# Parameters
SAMPLE_RATE = 16000  # Required sample rate for NeMo ASR models
DURATION = 2  # Duration of each audio chunk in seconds

# Predefined commands
commands = {"up": 2.0, "down":1.8, "left":1.5, "right":1.5}

# Set a similarity threshold for command recognition
SIMILARITY_THRESHOLD = 1.5  # Adjust between 0 (low) and 1 (strict)

def recognize_direction(snake_direction):
    print("Starting real-time voice recognition with adjustable thresholds...")

    def get_most_similar_command(transcription):
        """
        Find the most similar command to the transcription based on the similarity threshold.
        """
        best_match = None
        best_score = 0.0
        for command in commands:
            similarity = SequenceMatcher(None, transcription, command).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = command
        return best_match, best_score

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
            print(f"Recognized transcription: {transcription}")

            # Find the most similar command and check its similarity score
            best_match, best_score = get_most_similar_command(transcription)
            print(f"Best match: {best_match} (Score: {best_score})")

            # Accept the command if the similarity score exceeds the threshold
            if best_score >= SIMILARITY_THRESHOLD:
                print(f"Accepted command: {best_match}")
                if best_match == "up" and snake_direction.value != 2:  # Prevent reverse direction
                    snake_direction.value = 0
                elif best_match == "down" and snake_direction.value != 0:
                    snake_direction.value = 2
                elif best_match == "left" and snake_direction.value != 1:
                    snake_direction.value = 3
                elif best_match == "right" and snake_direction.value != 3:
                    snake_direction.value = 1
            else:
                print("No valid command recognized.")

        except Exception as e:
            print(f"Error in recognition: {e}")
        finally:
            # Clean up the temporary file
            os.remove(temp_filename)

        time.sleep(0.1)  # Small delay to prevent high CPU usage

if __name__ == "__main__":
    snake_direction = Value(ctypes.c_int, 1)  # Initial direction is "RIGHT"
    recognize_direction(snake_direction)
