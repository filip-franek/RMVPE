import os
import torch

from src import download_model, MIR1K, SAMPLE_RATE, WINDOW_LENGTH # to_local_average_cents


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hop_length = 20
model_path = "models/rmvpe.pt"
MODEL_DOWNLOAD_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true"
if not os.path.exists(model_path):
    print(f"Model {model_path} does not exist, downloading it.")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_model(MODEL_DOWNLOAD_URL, model_path)

model = torch.load(model_path, map_location=torch.device('cpu'))
# model.eval()
# model.module

dataset = MIR1K('datasets/MIR-1K', hop_length, None, ['test'])
for data in dataset:
    audio = data['audio'].to(device)
    pitch_label = data['pitch'].to(device)
    pitch_pred = torch.zeros_like(pitch_label)
    slices = range(0, pitch_pred.shape[0], 128)
    for start_steps in slices:
        end_steps = start_steps + 127
        start = int(start_steps * hop_length * SAMPLE_RATE / 1000)
        end = int(end_steps * hop_length * SAMPLE_RATE / 1000) + WINDOW_LENGTH
        if end_steps >= pitch_pred.shape[0]:
            t_audio = F.pad(audio[start:end], (0, end - start - len(audio[start:end])), mode='constant')
            t_pitch_pred = model(t_audio.reshape((-1, t_audio.shape[-1]))).squeeze(0)
            pitch_pred[start_steps:end_steps + 1] = t_pitch_pred[:pitch_pred.shape[0] - end_steps - 1]
        else:
            t_audio = audio[start:end]
            t_pitch_pred = model(t_audio.reshape((-1, t_audio.shape[-1]))).squeeze(0)
            pitch_pred[start_steps:end_steps + 1] = t_pitch_pred
