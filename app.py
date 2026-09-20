import streamlit as st
import torch
from audiocraft.models import MusicGen
import scipy.io.wavfile
import os

st.set_page_config(page_title="AI Music Generator", page_icon="🎵")

st.title("🎵 Meta MusicGen - AI Music Generator")
st.write("Create custom music tracks from text prompts using Meta's AudioCraft model right here!")

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('small')
    return model

with st.spinner("Loading AI Model... Please wait (this takes a moment)..."):
    model = load_model()

prompt = st.text_area("Enter your music description (e.g., '90s rock song with electric guitar and steady drums')", "Lo-fi hip hop beat with chill piano chords")
duration = st.slider("Duration (seconds)", min_value=5, max_value=30, value=10)

if st.button("Generate Music 🎶"):
    if not prompt.strip():
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating your music track... Sit back and relax!"):
            model.set_generation_parameters(duration=duration)
            wav = model.generate([prompt])  
            
            sample_rate = model.sample_rate
            wav_data = wav[0, 0].cpu().numpy()
            output_file = "generated_music.wav"
            scipy.io.wavfile.write(output_file, sample_rate, wav_data)
            
            st.success("Music generated successfully!")
            st.audio(output_file, format="audio/wav")
          
