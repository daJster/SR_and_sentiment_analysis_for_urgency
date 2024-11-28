import librosa
import numpy as np
import soundfile as sf

# Load audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Add noise to the audio
def add_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

# Apply time-stretching (speed change)
def time_stretch(audio, rate=1.2):
    return librosa.effects.time_stretch(audio, rate)


def extract_audio_features(file_path):
    # Load audio
    y, sr = librosa.load(file_path)
    
    # Extract pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return pitch, mfccs_mean



# Example usage
file_path = "example_audio.wav"
audio, sr = load_audio(file_path)

# Add noise
audio_noisy = add_noise(audio)

# Time-stretching
audio_stretched = time_stretch(audio)

# Save augmented audio files
sf.write("audio_noisy.wav", audio_noisy, sr)
sf.write("audio_stretched.wav", audio_stretched, sr)

# Example usage
audio_path = 'path/to/audio/file.wav'
transcript_text = "Please help! There's an emergency situation here."

