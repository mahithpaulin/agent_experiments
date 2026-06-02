import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def binary_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8), axis=1)

def kl_divergence(mu, logvar):
    return -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar), axis=1)

def initialize_weights(input_dim, latent_dim):
    W1 = np.random.randn(input_dim, latent_dim) * 0.01
    b1 = np.zeros(latent_dim)
    W2_mu = np.random.randn(latent_dim, latent_dim) * 0.01
    b2_mu = np.zeros(latent_dim)
    W2_logvar = np.random.randn(latent_dim, latent_dim) * 0.01
    b2_logvar = np.zeros(latent_dim)
    W3 = np.random.randn(latent_dim, input_dim) * 0.01
    b3 = np.zeros(input_dim)
    return W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar, W3, b3

def forward_encoder(X, W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar):
    hidden = tanh(np.dot(X, W1) + b1)
    mu = np.dot(hidden, W2_mu) + b2_mu
    logvar = np.dot(hidden, W2_logvar) + b2_logvar
    return mu, logvar

def reparameterize(mu, logvar):
    std = np.exp(0.5 * logvar)
    epsilon = np.random.randn(*std.shape)
    return mu + epsilon * std

def forward_decoder(z, W3, b3):
    return sigmoid(np.dot(z, W3) + b3)