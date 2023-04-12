#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import librosa
import torch
import torch.nn.functional as F
from models.bart_captioning import BartCaptionModel

checkpoint_path = ""
audio_path = ""
cp = torch.load(checkpoint_path)

config = cp["config"]
model = BartCaptionModel(config)
model.load_state_dict(cp["model"])
device = torch.device(config["device"])
model.to(device)

waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
waveform = torch.tensor(waveform)

if config["audio_encoder_args"]["model_arch"] == "transformer":
    max_length = 32000 * 10
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        waveform = F.pad(waveform, [0, max_length - len(waveform)], "constant", 0.0)

else:
    max_length = 32000 * 30
    if len(waveform) > max_length:
        waveform = waveform[:max_length]

waveform = waveform.unsqueeze(0)

model.eval()
with torch.no_grad():
    waveform = waveform.to(device)
    caption = model.generate(samples=waveform, num_beams=3)

print(caption)
