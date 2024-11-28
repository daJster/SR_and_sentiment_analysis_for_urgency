import streamlit as st
import torch
import librosa
import time
from io import BytesIO
from pydub import AudioSegment
from model import AudioUrgencyModel, text_classifier, S2T_model, S2T_processor, device
import numpy as np


import sys
sys.path.append('/home/jelkarchi/.homenv/lib/python3.10/site-packages/ffprobe')
sys.path.append('/home/jelkarchi/.homenv/lib/python3.10/site-packages/ffmpeg')

# Load your pre-trained models (replace with actual models)
audio_model = AudioUrgencyModel().to(device)

# Placeholder for audio recording (Streamlit doesn't natively support audio files from mic_recorder)
def save_audio(audio):
    with open("temp_audio.wav", "wb") as f:
        f.write(audio)
    return "temp_audio.wav"

# Urgency analysis function
def analyze_urgency(audio, transcript_text):
    # Extract audio features
    y, sr = librosa.load(audio, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Convert to tensor for model input
    audio_input = torch.tensor(mfcc_mean).unsqueeze(0).float()
    
    # Predict audio urgency
    audio_model.eval()
    with torch.no_grad():
        audio_urgency_score = audio_model(audio_input).item()
    
    # Predict text urgency using sentiment analysis
    text_result = text_classifier(transcript_text)[0]
    text_urgency_score = 1 if text_result['label'] == 'LABEL_1' else 0  # Adjust labels based on sentiment model output
    
    # Combine scores
    final_urgency_score = 0.6 * audio_urgency_score + 0.4 * text_urgency_score
    
    return final_urgency_score


def transcribe_audio(segment, model, processor, sampling_rate=16000):
    # Load audio file
    audio_data = np.array(segment).astype(np.float32) / 2**15
    
    # Process the audio
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")

    # Generate transcription (send data to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
    
    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    # Streamlit app logic
    st.title("Real-time Urgency Detection")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a"])
    sampling_rate = 16000  # Set the desired sampling rate
    segment_duration_ms = 120 * 1000    
    
    if uploaded_file:
        # Load and preprocess the audio file
        audio, sampling_rate = librosa.load(BytesIO(uploaded_file.getvalue()), sr=sampling_rate)

        # Divide audio into 3-second chunks
        if len(audio) > segment_duration_ms:
            num_segments = len(audio) // segment_duration_ms
        else :
            num_segments = 1
        
        # Display transcription in real-time
        text_container = st.empty()
        transcription = ""
        
        for i in range(num_segments):
            segment = audio[i * segment_duration_ms: (i + 1) * segment_duration_ms]
            transcription += " " + transcribe_audio(segment, S2T_model, S2T_processor, sampling_rate)
            text_container.write(f"{transcription}")