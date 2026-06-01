import pytest
import torch
from main import generate_2d_gaussian_clusters, VAE, loss_function, train_vae, DEVICE
import numpy as np

@pytest.fixture
def synthetic_data():
    data, labels = generate_2d_gaussian_clusters(n_samples=300, n_clusters=3, cluster_std=0.1, seed=123)
    return data, labels

def test_generate_2d_gaussian_clusters_shapes(synthetic_data):
    data, labels = synthetic_data
    assert data.shape == (300, 2)
    assert labels.shape == (300,)
    assert torch.all(labels < 3)
    assert torch.all(labels >= 0)

def test_vae_forward_shapes(synthetic_data):
    data, _ = synthetic_data
    model = VAE().to(DEVICE)
    recon, mu, logvar, z = model(data)
    assert recon.shape == data.shape
    assert mu.shape == (data.shape[0], 2)
    assert logvar.shape == (data.shape[0], 2)
    assert z.shape == (data.shape[0], 2)

def test_loss_function_values(synthetic_data):
    data, _ = synthetic_data
    model = VAE().to(DEVICE)
    recon, mu, logvar, _ = model(data)
    recon_loss, kld = loss_function(recon, data, mu, logvar)
    assert recon_loss.item() >= 0
    assert kld.item() >= 0

def test_reparameterization_variance():
    model = VAE().to(DEVICE)
    mu = torch.zeros(1000, 2).to(DEVICE)  # Increased sample size for better variance estimate
    logvar = torch.zeros(1000, 2).to(DEVICE)
    z = model.reparameterize(mu, logvar)
    # Since mu=0, logvar=0 => std=1, z should have variance close to 1
    var = z.var(dim=0)
    assert torch.allclose(var, torch.ones(2).to(DEVICE), atol=0.1)

def test_training_decreases_losses():
    data, labels = generate_2d_gaussian_clusters(n_samples=600, n_clusters=3, cluster_std=0.5, seed=42)
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    model = VAE().to(DEVICE)
    recon_losses, kld_losses = train_vae(model, dataloader, epochs=20, lr=1e-2)
    assert recon_losses[-1] < recon_losses[0]
    assert kld_losses[-1] < kld_losses[0]

def test_latent_space_separation():
    data, labels = generate_2d_gaussian_clusters(n_samples=300, n_clusters=3, cluster_std=0.3, seed=123)
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    model = VAE().to(DEVICE)
    train_vae(model, dataloader, epochs=30, lr=1e-3)
    model.eval()
    with torch.no_grad():
        _, mu, _, _ = model(data)
    mu = mu.cpu()
    labels = labels.cpu()
    # Compute mean latent vector per cluster
    means = []
    for c in range(3):
        cluster_mu = mu[labels == c]
        means.append(cluster_mu.mean(dim=0))
    means = torch.stack(means)
    # Check pairwise distances between cluster means are sufficiently large
    d01 = torch.norm(means[0] - means[1])
    d12 = torch.norm(means[1] - means[2])
    d02 = torch.norm(means[0] - means[2])
    assert d01 > 0.5
    assert d12 > 0.5
    assert d02 > 0.5
