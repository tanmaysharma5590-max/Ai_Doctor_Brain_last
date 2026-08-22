"""
brain_of_the_doctor.py
Handles image encoding and sends the image + a question to a Groq
multimodal (vision) model, returning the model's text reply.
"""

import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Groq's currently supported multimodal model (llama-4-scout was retired
# by Groq on 2026-06-17 for free/developer tiers; qwen3.6-27b replaced it).
VISION_MODEL = "qwen/qwen3.6-27b"


def encode_image(image_path: str) -> str:
    """Encode an image file on disk to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_bytes(image_bytes: bytes) -> str:
    """Encode raw image bytes (e.g. from a Streamlit uploader) to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_image_with_query(query: str, encoded_image: str, model: str = VISION_MODEL) -> str:
    """
    Send a base64-encoded image plus a text query to a Groq vision model
    and return the plain-text reply.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return chat_completion.choices[0].message.content