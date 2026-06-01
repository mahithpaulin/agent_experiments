import torch

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t).float()
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise, betas):
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_alphas_cumprod = extract(alphas_cumprod.sqrt(), t, x_start.shape)
    sqrt_one_minus_alphas_cumprod = extract((1 - alphas_cumprod).sqrt(), t, x_start.shape)
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

def p_losses(denoise_model, x_start, t, betas):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise, betas)
    noise_pred = denoise_model(x_noisy, t / betas.shape[0])
    return torch.nn.functional.mse_loss(noise_pred, noise)

def sample_timesteps(n, timesteps):
    return torch.randint(low=0, high=timesteps, size=(n,))