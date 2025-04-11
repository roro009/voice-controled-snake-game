import speech_recognition as sr

def transcribe_with_google(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio).lower()
    except (sr.UnknownValueError, sr.RequestError):
        return None
