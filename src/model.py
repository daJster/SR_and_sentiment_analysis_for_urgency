import librosa
import numpy as np
import torch
import torch.nn as nn
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import pipeline

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained text sentiment model and move to GPU if available
text_classifier = pipeline(
    'text-classification',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=0 if torch.cuda.is_available() else -1  # GPU index 0 or CPU fallback
)

S2T_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
S2T_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

# Define a simple neural network for audio feature classification
class AudioUrgencyModel(nn.Module):
    def __init__(self):
        super(AudioUrgencyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 128),  # Input size depends on extracted features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    # Example usage
    pass