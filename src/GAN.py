import librosa as lr
import numpy as np
from pathlib import Path
import IPython.display as ipd
import torch
import torchaudio
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class DrumDataset(Dataset):
    def __init__(self, file_paths, transform=None, target_length=16384):
        self.file_paths = file_paths
        self.transform = transform
        self.target_length = target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.file_paths[idx])
        audio = audio.mean(dim=0)
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            padding = self.target_length - len(audio)
            audio = torch.nn.functional.pad(audio, (0, padding))

        if self.transform:
            audio = self.transform(audio)

        # print(audio.shape)
        
        return audio.unsqueeze(0)

class WaveGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, model_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256*model_dim),
            nn.Unflatten(1, (16*model_dim, 16)),
            nn.ReLU(True),
            nn.ConvTranspose1d(16*model_dim, 8*model_dim, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(8*model_dim, 4*model_dim, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(4*model_dim, 2*model_dim, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(2*model_dim, model_dim, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(model_dim, 1, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # print("G Input shape:", x.shape)
        # for layer in self.network:
        #     x = layer(x)
        #     print(f"After {layer.__class__.__name__}:", x.shape)
        return self.network(x)


class WaveGANDiscriminator(nn.Module):
    def __init__(self, model_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, model_dim, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(model_dim, 2*model_dim, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(2*model_dim, 4*model_dim, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(4*model_dim, 8*model_dim, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8*model_dim, 16*model_dim, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(256*model_dim, 1),
        )

    def forward(self, x):
        # print("D Input shape:", x.shape)
        # for layer in self.network:
        #     x = layer(x)
        #     print(f"After {layer.__class__.__name__}:", x.shape)
        return self.network(x)

def gradient_penalty(critic, real_samples, fake_samples, device="cpu"):
    # Interpolate between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Get critic output for the interpolated samples
    interpolated_output = critic(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def critic_loss(real_output, fake_output):
    return torch.mean(fake_output) - torch.mean(real_output)

def generator_loss(fake_output):
    return -torch.mean(fake_output)

criterion = nn.BCELoss()
model_dim = 64
beta1 = 0.5

generator = WaveGANGenerator()
discriminator = WaveGANDiscriminator()

optimizerG = optim.Adam(generator.parameters(), lr=0.0001, betas=(beta1, 0.9))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(beta1, 0.9))


file_paths = [str(p) for p in Path().glob('../data/Kicks/*.wav')]

dataset = DrumDataset(file_paths)

batch_size = 5
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 30
latent_dim = 100
n_critic = 1
lambda_gp = 20
num_batches = 5

for epoch in range(num_epochs):
    for idx, real_data in enumerate(dataloader):
        if idx >= num_batches:
            break

        # Move real data to device and train discriminator (critic)
        real_data = real_data.to(device)
        for _ in range(n_critic):
            noise = torch.rand(real_data.size(0), latent_dim, device=device) * 2 - 1
            fake_samples = generator(noise).detach()
            
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_samples)
            
            gp = gradient_penalty(discriminator, real_data, fake_samples, device)
            d_loss = critic_loss(real_output, fake_output) + lambda_gp * gp
            
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

        # Train generator
        noise = torch.rand(real_data.size(0), latent_dim, device=device) * 2 - 1
        fake_data = generator(noise)
        g_loss = generator_loss(discriminator(fake_data))

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()
    print(f"Epoch: {epoch+1}/{num_epochs}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")
        

# Generate sample
noise = torch.rand(batch_size, latent_dim).to(device)*2 - 1

generator.eval()

with torch.no_grad():
    generated_sample = generator(noise).squeeze().cpu().numpy()

lr.display.waveshow(generated_sample)
# for sample in dataloader:
#     lr.display.waveshow(sample.numpy())

samples = [lr.load(p)[0] for p in Path().glob('../data/Snares/*.wav')]
sr = [lr.load(p)[1] for p in Path().glob('../data/Snares/*.wav')]

lr.display.waveshow(samples[5])