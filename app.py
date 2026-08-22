"""
app.py
AI Doctor — Streamlit front end.

Flow:
  1. Patient records/uploads a voice question and uploads a photo.
  2. voice_of_the_patient.py transcribes the question (Groq Whisper).
  3. brain_of_the_doctor.py sends the photo + question to a Groq vision model.
  4. voice_of_the_doctor.py turns the doctor's text reply into speech.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from brain_of_the_doctor import encode_image_bytes, analyze_image_with_query
from voice_of_the_patient import save_audio_bytes_as_mp3, transcribe_with_groq
from voice_of_the_doctor import generate_doctor_voice

load_dotenv()

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Doctor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = (
    "You have to act as a professional doctor, this is only for educational "
    "purposes. What's in this image? Do you find anything concerning medically? "
    "If you form a differential, suggest a few simple remedies. Don't use numbers "
    "or special characters or markdown. Answer in one short paragraph, no more "
    "than four sentences, as if you are speaking directly to the patient in "
    "person. Start with something like 'With what I see, I think you have...' "
    "instead of describing the image. No preamble, start your answer right away."
)

DEFAULT_QUERY = "Is there something wrong with my skin?"

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at 10% 0%, #eef6ff 0%, #f7fbff 35%, #ffffff 100%);
        }

        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 28px 34px;
            border-radius: 20px;
            background: linear-gradient(120deg, #0f766e 0%, #14b8a6 55%, #38bdf8 100%);
            box-shadow: 0 12px 30px rgba(15, 118, 110, 0.25);
            margin-bottom: 28px;
        }
        .hero-left {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .hero-icon {
            font-size: 46px;
            line-height: 1;
        }
        .hero-title {
            color: #ffffff;
            font-size: 30px;
            font-weight: 700;
            margin: 0;
        }
        .hero-subtitle {
            color: #e6fffb;
            font-size: 15px;
            margin: 4px 0 0 0;
            opacity: 0.95;
        }
        .hero-credit {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.35);
            color: #ffffff;
            font-size: 13px;
            font-weight: 600;
            white-space: nowrap;
            backdrop-filter: blur(4px);
        }

        .panel {
            background: #ffffff;
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.15);
            height: 100%;
        }
        .panel h3 {
            margin-top: 0;
            font-size: 18px;
            color: #0f172a;
        }
        .panel-caption {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 14px;
        }

        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #14b8a6;
            color: white;
            font-size: 13px;
            font-weight: 600;
            margin-right: 8px;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(120deg, #0f766e, #14b8a6);
            color: white;
            font-weight: 600;
            padding: 0.6em 0;
            border: none;
            box-shadow: 0 6px 16px rgba(20, 184, 166, 0.35);
            transition: transform 0.15s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            background: linear-gradient(120deg, #0d9488, #22d3ee);
        }

        .result-card {
            background: linear-gradient(135deg, #ecfeff 0%, #f0fdfa 100%);
            border-left: 4px solid #14b8a6;
            border-radius: 14px;
            padding: 18px 20px;
            margin-top: 14px;
        }
        .result-label {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #0f766e;
            margin-bottom: 6px;
        }
        .result-text {
            font-size: 15.5px;
            color: #0f172a;
            line-height: 1.55;
        }

        .disclaimer {
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
            margin-top: 30px;
        }

        .credit {
            font-size: 12.5px;
            color: #0f766e;
            text-align: center;
            font-weight: 600;
            margin-top: 34px;
            letter-spacing: 0.02em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-left">
            <div class="hero-icon">🩺</div>
            <div>
                <p class="hero-title">AI Doctor — with Vision &amp; Voice</p>
                <p class="hero-subtitle">Speak your symptom, show a photo, and get a spoken response — instantly.</p>
            </div>
        </div>
        <div class="hero-credit">👨‍💻 Made by Tanmay Sharma</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ GROQ_API_KEY is not set. Add it to a `.env` file next to app.py before running a consultation.")

# ----------------------------------------------------------------------
# Input panels
# ----------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-badge">1</span>Describe your symptom</h3>', unsafe_allow_html=True)
    st.markdown('<p class="panel-caption">Record your question with your microphone, or type it instead.</p>', unsafe_allow_html=True)

    audio_value = st.audio_input("Record your question")
    typed_query = st.text_input("...or type your question", placeholder=DEFAULT_QUERY)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h3><span class="step-badge">2</span>Show the doctor a photo</h3>', unsafe_allow_html=True)
    st.markdown('<p class="panel-caption">Upload a clear, well-lit photo of the affected area.</p>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
run = st.button("🔎  Get the doctor's opinion", use_container_width=True)

# ----------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------
if run:
    if not uploaded_image:
        st.error("Please upload a photo before requesting a consultation.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing — add it to your .env file first.")
    else:
        with st.spinner("Listening..."):
            patient_text = typed_query.strip()
            if not patient_text and audio_value is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    save_audio_bytes_as_mp3(audio_value.read(), tmp_audio.name)
                    patient_text = transcribe_with_groq(tmp_audio.name)
                os.unlink(tmp_audio.name)
            if not patient_text:
                patient_text = DEFAULT_QUERY

        with st.spinner("Examining the image..."):
            encoded_image = encode_image_bytes(uploaded_image.getvalue())
            full_query = SYSTEM_PROMPT + "\n\nPatient's question: " + patient_text
            doctor_reply = analyze_image_with_query(full_query, encoded_image)

        with st.spinner("Preparing the doctor's voice..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_reply:
                reply_audio_path = generate_doctor_voice(doctor_reply, tmp_reply.name)

        st.markdown("---")
        result_col1, result_col2 = st.columns(2, gap="large")

        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">🗣️ What we heard</div>
                    <div class="result-text">{patient_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">🩺 Doctor's response</div>
                    <div class="result-text">{doctor_reply}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.audio(reply_audio_path, format="audio/mp3")

        os.unlink(reply_audio_path)

st.markdown(
    '<p class="disclaimer">This tool is for educational and demo purposes only and is not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="credit">🩺 Made by Tanmay Sharma</p>',
    unsafe_allow_html=True,
)