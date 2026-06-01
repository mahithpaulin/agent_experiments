import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t).float()
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise, betas):
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_alphas_cumprod = torch.sqrt(extract(alphas_cumprod, t, x_start.shape))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - extract(alphas_cumprod, t, x_start.shape))
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = DoubleConv(1, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 96)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dconv_up2 = DoubleConv(96 + 64, 64)
        self.dconv_up1 = DoubleConv(64 + 32, 32)
        self.conv_last = nn.Conv2d(32, 1, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 96)
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()
        temb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
        conv1 = self.dconv_down1(x)
        conv1 = conv1 + temb[:, :32, :, :]
        x1 = self.maxpool(conv1)
        conv2 = self.dconv_down2(x1)
        conv2 = conv2 + temb[:, 32:96, :, :][:, :64, :, :]
        x2 = self.maxpool(conv2)
        x3 = self.dconv_down3(x2)
        # Fix: slice temb correctly for x3 addition
        x3 = x3 + temb[:, 64:96, :, :]
        x = self.upsample(x3)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = x + temb[:, 32:64, :, :]
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = x + temb[:, :32, :, :]
        out = self.conv_last(x)
        return out

def train(steps=1000, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    betas = cosine_beta_schedule(steps).to(device)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        x = torch.randn(16, 1, 16, 16, device=device)
        t = torch.randint(0, steps, (16,), device=device)
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, noise, betas)
        noise_pred = model(x_noisy, t.unsqueeze(-1).float() / steps)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, betas

def sample(model, betas, n=4, steps=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    x = torch.randn(n, 1, 16, 16, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    with torch.no_grad():
        for i in reversed(range(steps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            noise_pred = model(x, t.unsqueeze(-1).float() / steps)
            alpha = alphas[i]
            alpha_cum = alphas_cumprod[i]
            beta = betas[i]
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_cum).sqrt()) * noise_pred) + beta.sqrt() * noise
    return x

def plot_denoising_trajectory(model, betas, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    n = 1
    steps = len(betas)
    x = torch.randn(n, 1, 16, 16, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    imgs = []
    with torch.no_grad():
        for i in reversed(range(steps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            noise_pred = model(x, t.unsqueeze(-1).float() / steps)
            alpha = alphas[i]
            alpha_cum = alphas_cumprod[i]
            beta = betas[i]
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_cum).sqrt()) * noise_pred) + beta.sqrt() * noise
            if i % (steps // 10) == 0 or i == 0:
                imgs.append(x[0, 0].cpu())
    fig, axs = plt.subplots(1, len(imgs), figsize=(15, 2))
    for ax, img in zip(axs, imgs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    model, betas = train(steps=1000, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    plot_denoising_trajectory(model, betas)
    samples = sample(model, betas, n=4)
    for i, sample_img in enumerate(samples):
        plt.subplot(1, 4, i+1)
        plt.imshow(sample_img[0].cpu(), cmap='gray')
        plt.axis('off')
    plt.show()
