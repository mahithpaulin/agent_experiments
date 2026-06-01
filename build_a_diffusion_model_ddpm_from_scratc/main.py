import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # t must be long tensor
        if t.dtype != torch.long:
            t = t.long()
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        if t.dtype != torch.long:
            t = t.long()
        sqrt_recip_alphas_cumprod_t = (1 / self.sqrt_alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = (torch.sqrt(1 / self.alphas_cumprod[t] - 1)).view(-1, 1, 1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def p_sample(self, model, x_t, t):
        if t.dtype != torch.long:
            t = t.long()
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1 / torch.sqrt(self.alphas[t])).view(-1, 1, 1, 1)

        # Pass t as float for embedding
        t_float = t.float()
        noise_pred = model(x_t, t_float)
        x0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
        x0_pred = torch.clamp(x0_pred, -1., 1.)

        posterior_mean = (
            betas_t * x0_pred + (1 - betas_t) * x_t
        ) / torch.sqrt(1 - self.betas[t])

        if t[0] == 0:
            return posterior_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_var = self.posterior_variance[t].view(-1, 1, 1, 1)
            return posterior_mean + torch.sqrt(posterior_var) * noise


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t should be float tensor
        if t.dtype != torch.float:
            t = t.float()
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x, t):
        h = self.conv(x)
        time_emb = self.time_mlp(t).view(t.size(0), -1, 1, 1)
        return h + time_emb


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, time_emb_dim)

    def forward(self, x, t):
        x = self.pool(x)
        return self.conv(x, t)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, time_emb_dim)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)


class UNet(nn.Module):
    def __init__(self, time_emb_dim=32):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.inc = DoubleConv(1, 64, time_emb_dim)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 256, time_emb_dim)
        self.up1 = Up(256, 128, time_emb_dim)
        self.up2 = Up(128, 64, time_emb_dim)
        self.up3 = Up(64, 64, time_emb_dim)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        # t should be float tensor for embedding
        if t.dtype != torch.float:
            t_emb = self.time_embedding(t.float())
        else:
            t_emb = self.time_embedding(t)
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x = self.up1(x4, x3, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.up3(x, x1, t_emb)
        return self.outc(x)


class CircleDataset(Dataset):
    def __init__(self, size=1000, img_size=32):
        self.size = size
        self.img_size = img_size
        self.data = self.generate_circles(size, img_size)

    def generate_circles(self, n, img_size):
        imgs = []
        for _ in range(n):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            radius = np.random.uniform(img_size * 0.1, img_size * 0.4)
            x0 = np.random.uniform(radius, img_size - radius)
            y0 = np.random.uniform(radius, img_size - radius)
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2
            img[mask] = 1.0
            imgs.append(img)
        imgs = np.array(imgs)[:, None, :, :]
        return torch.tensor(imgs)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = Diffusion(timesteps=1000, device=device)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = CircleDataset(size=1000, img_size=32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.size(0),), device=device, dtype=torch.long)
            noise = torch.randn_like(batch)
            x_noisy = diffusion.q_sample(batch, t, noise)
            # Pass t as float for model
            noise_pred = model(x_noisy, t.float())
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig('training_loss.png')
    plt.show()

    return diffusion, model, dataset


def sample(diffusion, model, shape, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        imgs = []
        for i in reversed(range(diffusion.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            # Pass t as float for model
            x = diffusion.p_sample(model, x, t)
            if i % (diffusion.timesteps // 10) == 0:
                imgs.append(x.cpu().clone())
        return x.cpu(), imgs


def plot_denoising_trajectory(imgs):
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    for i, img in enumerate(imgs):
        axs[i].imshow(img[0, 0], cmap='gray', vmin=-1, vmax=1)
        axs[i].axis('off')
        axs[i].set_title(f"t={i * (1000 // n)}")
    plt.tight_layout()
    plt.savefig('denoising_trajectory.png')
    plt.show()


if __name__ == "__main__":
    diffusion, model, dataset = train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = (1, 1, 32, 32)
    sample_img, trajectory = sample(diffusion, model, shape, device)
    plot_denoising_trajectory(trajectory)
