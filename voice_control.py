import sounddevice as sd
import numpy as np
import tempfile
import wave
import os
import time
from multiprocessing import Value
import ctypes
from difflib import SequenceMatcher

from utils.asr_google import transcribe_with_google
from utils.asr_nemo import transcribe_with_nemo

SAMPLE_RATE = 16000
DURATION = 2
commands = ["up", "down", "left", "right"]
SIMILARITY_THRESHOLD = 0.7

def get_most_similar_command(text):
    best_match, best_score = None, 0
    for cmd in commands:
        score = SequenceMatcher(None, text, cmd).ratio()
        if score > best_score:
            best_match = cmd
            best_score = score
    return best_match, best_score

def recognize_direction(snake_direction):
    while True:
        print("Listening...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        text = transcribe_with_google(temp_path)
        if not text:
            print("Google ASR failed. Falling back to NeMo.")
            text = transcribe_with_nemo(temp_path)

        os.remove(temp_path)

        if text:
            match, score = get_most_similar_command(text)
            if score >= SIMILARITY_THRESHOLD:
                print(f"Command: {match}")
                if match == "up" and snake_direction.value != 2:
                    snake_direction.value = 0
                elif match == "down" and snake_direction.value != 0:
                    snake_direction.value = 2
                elif match == "left" and snake_direction.value != 1:
                    snake_direction.value = 3
                elif match == "right" and snake_direction.value != 3:
                    snake_direction.value = 1
            else:
                print("No valid match.")
        else:
            print("ASR failed.")

        time.sleep(0.1)
