"""
voice_of_the_doctor.py
Converts the doctor's text reply into speech. Uses ElevenLabs when an API
key is available (better quality), and always falls back to gTTS (free,
no key required) so the app keeps working either way.
"""

import os
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()

ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")


def text_to_speech_with_gtts(input_text: str, output_filepath: str) -> str:
    """Convert text to speech with gTTS and save it as an mp3 file."""
    audioobj = gTTS(text=input_text, lang="en", slow=False)
    audioobj.save(output_filepath)
    return output_filepath


def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str) -> str:
    """Convert text to speech with ElevenLabs and save it as an mp3 file."""
    import elevenlabs
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2",
    )
    elevenlabs.save(audio, output_filepath)
    return output_filepath


def generate_doctor_voice(input_text: str, output_filepath: str = "final_doctor_response.mp3") -> str:
    """
    Generate the doctor's spoken reply. Prefers ElevenLabs if a key is
    configured; silently falls back to gTTS on any failure.
    """
    if ELEVENLABS_API_KEY:
        try:
            return text_to_speech_with_elevenlabs(input_text, output_filepath)
        except Exception:
            pass
    return text_to_speech_with_gtts(input_text, output_filepath)