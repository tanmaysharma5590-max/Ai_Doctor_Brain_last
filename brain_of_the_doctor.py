"""
brain_of_the_doctor.py
Handles image encoding and sends the image + a question to a Groq
multimodal (vision) model, returning the model's text reply.
"""

import os
import base64
from dotenv import load_dotenv
from groq import Groq, NotFoundError

load_dotenv()


def _get_groq_api_key() -> str | None:
    """Read the key from the environment, with a Streamlit Cloud secrets fallback."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


GROQ_API_KEY = _get_groq_api_key()

# Pehla model try hoga, agar Groq ne hata diya ho toh agla try hoga.
# Groq models list: https://console.groq.com/docs/models
VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
]
VISION_MODEL = VISION_MODELS[0]


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
    and return the plain-text reply. Model not found hone par next model try karta hai.
    """
    api_key = GROQ_API_KEY or _get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in your .env file locally, "
            "or in Settings -> Secrets on Streamlit Cloud."
        )

    client = Groq(api_key=api_key)

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

    # Diya hua model pehle, phir baaki fallback models (duplicate hata ke)
    candidates = [model] + [m for m in VISION_MODELS if m != model]

    last_error = None
    for candidate in candidates:
        try:
            chat_completion = client.chat.completions.create(
                model=candidate,
                messages=messages,
            )
            return chat_completion.choices[0].message.content
        except NotFoundError as e:
            last_error = e
            continue

    # Koi bhi model nahi mila: account pe jo models available hain unki list dikhao
    try:
        available = [m.id for m in client.models.list().data]
    except Exception:
        available = []

    raise RuntimeError(
        f"Koi vision model available nahi mila (tried: {candidates}). "
        f"Groq pe available models: {available}. Last error: {last_error}"
    )
