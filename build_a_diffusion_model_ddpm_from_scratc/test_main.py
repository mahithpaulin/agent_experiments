import torch
import pytest
from main import UNet, train, sample
from utils import cosine_beta_schedule, q_sample, extract

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_cosine_beta_schedule():
    betas = cosine_beta_schedule(1000)
    assert betas.shape[0] == 1000
    assert (betas >= 0).all() and (betas <= 1).all()

def test_extract_shapes():
    a = torch.arange(10).float()
    t = torch.tensor([1, 3, 5])
    x_shape = (3, 2, 2)
    out = extract(a, t, x_shape)
    assert out.shape == (3, 1, 1)

def test_q_sample_consistency():
    betas = cosine_beta_schedule(1000)
    x_start = torch.randn(4,1,16,16)
    t = torch.tensor([0, 500, 999, 999])
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise, betas)
    assert x_noisy.shape == x_start.shape

@pytest.mark.parametrize("steps", [100, 500, 1000])
def test_train_loss_decreasing(steps):
    betas = cosine_beta_schedule(steps).to(DEVICE)
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(100):
        x = torch.randn(16,1,16,16, device=DEVICE)
        t = torch.randint(0, steps, (16,), device=DEVICE)
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, noise, betas)
        noise_pred = model(x_noisy, t/steps)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

def test_sample_reconstruction():
    model, betas = train()
    imgs = sample(model, betas, n=4)
    assert imgs.shape == (4,1,16,16)
    # Check values roughly in expected range with some tolerance
    assert imgs.min().item() >= pytest.approx(-10, abs=5)
    assert imgs.max().item() <= pytest.approx(10, abs=5)

def test_denoising_trajectory_plot():
    model, betas = train()
    model.eval()
    n = 1
    x = torch.randn(n,1,16,16, device=DEVICE)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    imgs = []
    for i in reversed(range(1000)):
        t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
        noise_pred = model(x, t/1000)
        alpha = alphas[i]
        alpha_cum = alphas_cumprod[i]
        beta = betas[i]
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        x = (1/alpha.sqrt())*(x - ((1 - alpha)/((1 - alpha_cum).sqrt()))*noise_pred) + beta.sqrt()*noise
        if i % 200 == 0:
            imgs.append(x.cpu().squeeze().detach())
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(imgs), figsize=(12,2))
    for i, img in enumerate(imgs):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.savefig('denoising_trajectory.png')
    plt.close()
