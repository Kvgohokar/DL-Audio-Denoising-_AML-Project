# predict.py
import torch
import torchaudio
import torchaudio.transforms as T
from clean import Generator  # Adjust path if needed

def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    waveform = waveform / waveform.abs().max()
    return waveform.squeeze(0), target_sr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denoise_audio(model, noisy_tensor):
    model.eval()
    model = model.to(device)
    noisy_tensor = noisy_tensor.to(device)
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0).unsqueeze(0))  # [1, 1, T]
        return output.squeeze().cpu().clamp(-1, 1)

def denoise(input_path, model_path, output_path):
    waveform, sr = load_audio(input_path)
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    enhanced = denoise_audio(model, waveform)
    torchaudio.save(output_path, enhanced.unsqueeze(0), sr)
    return output_path
