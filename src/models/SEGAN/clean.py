import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import torch.nn as nn
import torch.nn.functional as F

#####################################################################################
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, 15, stride=1, padding=7),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
#######################################################################################


def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Normalize to [-1, 1]
    waveform = waveform / waveform.abs().max()

    return waveform.squeeze(0), target_sr  # [T], sr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# noisy_tensor = noisy_tensor.to(device)

def denoise_audio(model, noisy_tensor):
    model.eval()
    with torch.no_grad():
        output = model(noisy_tensor.unsqueeze(0).unsqueeze(0))  # [1, 1, T]
        return output.squeeze().cpu().clamp(-1, 1)

def main(input_path, output_path, model_path):
    # Load and preprocess noisy audio
    noisy_tensor, sr = load_audio(input_path)

    # Load trained generator model
    #device = torch.device("cpu")  # or 'cuda' if available
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Enhance
    enhanced_tensor = denoise_audio(model, noisy_tensor)

    # Save result
    torchaudio.save(output_path, enhanced_tensor.unsqueeze(0), sr)
    print(f"Enhanced audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance audio using SEGAN Generator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input noisy .wav file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save enhanced .wav file")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to trained SEGAN generator .pth file")

    args = parser.parse_args()
    main(args.input, args.output, args.model)
