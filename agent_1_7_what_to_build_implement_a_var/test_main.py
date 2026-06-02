import numpy as np
import pytest
from utils import binary_cross_entropy, kl_divergence, initialize_weights, forward_encoder, reparameterize, forward_decoder
from main import generate_synthetic_data, train_vae

def test_generate_synthetic_data():
    data = generate_synthetic_data(n_samples=1000)
    assert data.shape == (1000, 2)
    assert np.isfinite(data).all()

def test_initialize_weights():
    W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar, W3, b3 = initialize_weights(2, 2)
    assert W1.shape == (2, 2)
    assert b1.shape == (2,)
    assert W2_mu.shape == (2, 2)
    assert b2_mu.shape == (2,)
    assert W2_logvar.shape == (2, 2)
    assert b2_logvar.shape == (2,)
    assert W3.shape == (2, 2)
    assert b3.shape == (2,)

def test_forward_encoder():
    X = np.random.randn(100, 2)
    W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar, _, _ = initialize_weights(2, 2)
    mu, logvar = forward_encoder(X, W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar)
    assert mu.shape == (100, 2)
    assert logvar.shape == (100, 2)

def test_reparameterize():
    mu = np.random.randn(100, 2)
    logvar = np.random.randn(100, 2)
    z = reparameterize(mu, logvar)
    assert z.shape == (100, 2)

def test_forward_decoder():
    z = np.random.randn(100, 2)
    _, _, _, _, _, _, W3, b3 = initialize_weights(2, 2)
    reconstruction = forward_decoder(z, W3, b3)
    assert reconstruction.shape == (100, 2)
    assert np.all(reconstruction >= 0) and np.all(reconstruction <= 1)

def test_binary_cross_entropy():
    y_true = np.random.randint(0, 2, (100, 2))
    y_pred = np.random.rand(100, 2)
    loss = binary_cross_entropy(y_true, y_pred)
    assert loss.shape == (100,)
    assert np.isfinite(loss).all()

def test_kl_divergence():
    mu = np.random.randn(100, 2)
    logvar = np.random.randn(100, 2)
    kl_loss = kl_divergence(mu, logvar)
    assert kl_loss.shape == (100,)
    assert np.isfinite(kl_loss).all()

def test_train_vae():
    data = generate_synthetic_data(n_samples=1000)
    latent_dim = 2
    mu, losses = train_vae(data, latent_dim=latent_dim, epochs=10)
    assert mu.shape == (1000, latent_dim)
    assert len(losses) == 10
    assert all(np.isfinite(loss) for loss in losses)