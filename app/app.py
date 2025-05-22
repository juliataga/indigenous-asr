import streamlit as st
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianTokenizer, MarianMTModel

st.set_page_config(page_title="Tribal ASR + Translation", layout="wide")
st.title("🗣️ Indigenous Language Transcription & Translation")

# Load models
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval()

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi").eval()
    return processor, model, tokenizer, translator

processor, asr_model, trans_tokenizer, trans_model = load_models()

# Upload audio
uploaded_audio = st.file_uploader("🎧 Upload audio (.wav)", type=["wav"])
if uploaded_audio:
    st.audio(uploaded_audio)

    waveform, sr = torchaudio.load(uploaded_audio)
    with st.spinner("Transcribing..."):
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = asr_model.generate(input_features)
        tribal_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    st.success("Transcription complete!")
    st.markdown(f"### 📝 Tribal Transcript:\n`{tribal_text}`")

    with st.spinner("Translating..."):
        tokens = trans_tokenizer.prepare_seq2seq_batch([tribal_text], return_tensors="pt")
        translated_ids = trans_model.generate(**tokens)
        translation = trans_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
    st.markdown("### 🌐 Hindi/English Translation:")
    st.success(translation)
