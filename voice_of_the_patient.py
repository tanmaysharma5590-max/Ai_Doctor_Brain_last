"""
voice_of_the_patient.py
Converts the patient's recorded audio into text using Groq's Whisper model.
Designed to work with bytes coming straight from Streamlit's audio input
widget (no local microphone / ffmpeg path juggling required).
"""

import logging
import os
import shutil
from io import BytesIO

from dotenv import load_dotenv
from pydub import AudioSegment
from groq import Groq

load_dotenv()

# Try to auto-locate ffmpeg on PATH instead of hardcoding a machine-specific
# path, so this works on Windows, macOS, Linux and in the cloud alike.
_ffmpeg_path = shutil.which("ffmpeg")
if _ffmpeg_path:
    AudioSegment.converter = _ffmpeg_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
STT_MODEL = "whisper-large-v3"


def save_audio_bytes_as_mp3(audio_bytes: bytes, output_filepath: str) -> str:
    """Take raw recorded audio bytes (wav/webm/etc.) and save them as an mp3 file."""
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
    audio_segment.export(output_filepath, format="mp3", bitrate="128k")
    logging.info(f"Audio saved to {output_filepath}")
    return output_filepath


def transcribe_with_groq(audio_filepath: str, stt_model: str = STT_MODEL, api_key: str = None) -> str:
    """Transcribe an audio file on disk to text using Groq's Whisper model."""
    api_key = api_key or GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    client = Groq(api_key=api_key)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en",
        )
    return transcription.text


if __name__ == "__main__":
    # Simple manual test: point this at any existing audio file.
    test_path = "patient_voice_test.mp3"
    if os.path.exists(test_path):
        print(transcribe_with_groq(test_path))
    else:
        print(f"No test file found at {test_path}. Run the Streamlit app instead.")