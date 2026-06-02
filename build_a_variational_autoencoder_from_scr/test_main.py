import numpy as np
import torch
from main import VAE, loss_function
from utils import generate_gaussian_data
import os

def test_vae_training():
    input_dim = 2
    latent_dim = 2
    epochs = 50
    batch_size = 128
    learning_rate = 0.001

    data = generate_gaussian_data(1000)
    data = torch.tensor(data, dtype=torch.float32)

    vae = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    kl_divs = []
    recon_losses = []

    for epoch in range(epochs):
        perm = torch.randperm(data.size(0))
        data = data[perm]
        epoch_kl = 0
        epoch_recon = 0

        for i in range(0, data.size(0), batch_size):
            batch = data[i:i+batch_size]
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss, recon_loss, kl_div = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_kl += kl_div.item()
            epoch_recon += recon_loss.item()

        kl_divs.append(epoch_kl / len(data))
        recon_losses.append(epoch_recon / len(data))

    torch.save(vae.state_dict(), "vae_model.pth")

    assert kl_divs[-1] < kl_divs[0], "KL divergence did not decrease"
    assert recon_losses[-1] < recon_losses[0], "Reconstruction loss did not decrease"

def test_latent_space_separation():
    input_dim = 2
    latent_dim = 2
    data = generate_gaussian_data(1000)
    data = torch.tensor(data, dtype=torch.float32)

    model_path = "vae_model.pth"
    assert os.path.exists(model_path), f"Model file '{model_path}' not found. Ensure the model is saved correctly."

    vae = VAE(input_dim, latent_dim)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()

    with torch.no_grad():
        mu, _ = vae.encode(data)
        mu = mu.numpy()

    assert mu.shape[1] == latent_dim, "Latent space dimensions are incorrect"
    cluster_centers = [(-5, -5), (5, 5), (-5, 5), (5, -5)]
    for center in cluster_centers:
        distances = np.linalg.norm(mu - np.array(center), axis=1)
        assert np.any(distances < 5.0), f"Cluster near {center} not well-separated"
