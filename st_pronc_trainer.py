import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from gtts import gTTS
import tempfile
import io
import soundfile as sf

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Pronunciation Trainer", layout="wide")

# =========================
# 🌸 Simple Pastel Theme
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E6F6FF, #FFC7ED);
}

.pastel-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 10px;
}

.stButton>button {
    background-color: #f9a8d4;
    color: white;
    border-radius: 20px;
    border: none;
}

.stProgress > div > div {
    background-color: #c084fc;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pastel-title">
    🌸 Pronunciation Trainer
</div>
""", unsafe_allow_html=True)

# =========================
# Generate reference audio
# =========================
def generate_reference_audio(text):
    tts = gTTS(text=text, lang='en')
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    y, sr = librosa.load(tmp_file.name, sr=22050)
    return y, sr

# =========================
# Calculate similarity
# =========================
def calculate_similarity(y_ref, y_user, sr):
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr, n_mfcc=13)

    min_len = min(mfcc_ref.shape[1], mfcc_user.shape[1])
    mfcc_ref = mfcc_ref[:, :min_len]
    mfcc_user = mfcc_user[:, :min_len]

    ref_vec = mfcc_ref.flatten()
    user_vec = mfcc_user.flatten()

    ref_vec /= np.linalg.norm(ref_vec)
    user_vec /= np.linalg.norm(user_vec)

    similarity = np.dot(ref_vec, user_vec)
    similarity = np.clip(similarity, 0, 1)

    return similarity

# =========================
# Plot waveform
# =========================
def plot_waveform(y, sr, title):
    fig, ax = plt.subplots(figsize=(6,2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# =========================
# Plot spectrogram
# =========================
def plot_spectrogram(y, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(6,2.5))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# =========================
# UI
# =========================
text_input = st.text_input("Type something to practice:")

if text_input:

    col1, col2 = st.columns(2)

    # 🔊 Reference
    with col1:
        st.subheader("🔊 Reference Voice")
        y_ref, sr = generate_reference_audio(text_input)

        ref_buffer = io.BytesIO()
        sf.write(ref_buffer, y_ref, sr, format='WAV')
        ref_buffer.seek(0)
        st.audio(ref_buffer)

        plot_waveform(y_ref, sr, "Reference Waveform")
        plot_spectrogram(y_ref, sr, "Reference Spectrogram")

    # 🎙 User
    with col2:
        st.subheader("🎙 Your Voice")
        user_audio = st.audio_input("Record here")

        if user_audio is not None:
            y_user, sr_user = librosa.load(user_audio, sr=22050)

            st.audio(user_audio)

            plot_waveform(y_user, sr_user, "Your Waveform")
            plot_spectrogram(y_user, sr_user, "Your Spectrogram")

            similarity = calculate_similarity(y_ref, y_user, sr)
            score = int(similarity * 100)

            st.subheader("📊 Result")
            st.progress(score / 100)
            st.metric("Pronunciation Score", f"{score} %")

# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from gtts import gTTS
# import tempfile
# import io
# import soundfile as sf

# st.set_page_config(page_title="Pronunciation Trainer", layout="wide")

# # =========================
# # Generate reference audio
# # =========================
# def generate_reference_audio(text):
#     tts = gTTS(text=text, lang='en')
#     tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(tmp_file.name)
#     y, sr = librosa.load(tmp_file.name, sr=22050)
#     return y, sr

# # =========================
# # Similarity (MFCC cosine)
# # =========================
# def calculate_similarity(y_ref, y_user, sr):
#     mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
#     mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr, n_mfcc=13)

#     min_len = min(mfcc_ref.shape[1], mfcc_user.shape[1])
#     mfcc_ref = mfcc_ref[:, :min_len]
#     mfcc_user = mfcc_user[:, :min_len]

#     ref_vec = mfcc_ref.flatten()
#     user_vec = mfcc_user.flatten()

#     ref_vec /= np.linalg.norm(ref_vec)
#     user_vec /= np.linalg.norm(user_vec)

#     similarity = np.dot(ref_vec, user_vec)
#     similarity = np.clip(similarity, 0, 1)

#     return similarity

# # =========================
# # Plot waveform
# # =========================
# def plot_waveform(y, sr, title):
#     fig, ax = plt.subplots(figsize=(8,3))
#     librosa.display.waveshow(y, sr=sr, ax=ax)
#     ax.set_title(title)
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Amplitude")
#     st.pyplot(fig)

# # =========================
# # Plot spectrogram
# # =========================
# def plot_spectrogram(y, sr, title):
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#     fig, ax = plt.subplots(figsize=(8,3))
#     img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
#     ax.set_title(title)
#     fig.colorbar(img, ax=ax, format="%+2.0f dB")
#     st.pyplot(fig)

# # =========================
# # UI
# # =========================
# st.title("🎤 Pronunciation Trainer")
# st.markdown("Practice speaking and compare your voice with reference.")

# text_input = st.text_input("Type something to practice:")

# if text_input:

#     col1, col2 = st.columns(2)

#     # ===== Reference =====
#     with col1:
#         st.subheader("🔊 Reference Voice")
#         y_ref, sr = generate_reference_audio(text_input)

#         ref_buffer = io.BytesIO()
#         sf.write(ref_buffer, y_ref, sr, format='WAV')
#         ref_buffer.seek(0)
#         st.audio(ref_buffer)

#         plot_waveform(y_ref, sr, "Reference Waveform")
#         plot_spectrogram(y_ref, sr, "Reference Spectrogram")

#     # ===== User =====
#     with col2:
#         st.subheader("🎙 Your Voice")
#         user_audio = st.audio_input("Record here")

#         if user_audio is not None:

#             y_user, sr_user = librosa.load(user_audio, sr=22050)

#             st.audio(user_audio)

#             plot_waveform(y_user, sr_user, "Your Waveform")
#             plot_spectrogram(y_user, sr_user, "Your Spectrogram")

#             similarity = calculate_similarity(y_ref, y_user, sr)
#             score = int(similarity * 100)

#             st.subheader("📊 Result")

#             # 🎯 Score bar
#             st.progress(score / 100)

#             st.metric("Pronunciation Score", f"{score} %")

#             # 📈 Comparison graph
#             fig, ax = plt.subplots()
#             ax.bar(["Reference", "Your Pronunciation"], [100, score])
#             ax.set_ylim(0, 100)
#             ax.set_ylabel("Similarity %")
#             ax.set_title("Pronunciation Comparison")
#             st.pyplot(fig)






