import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=1000):
    mean1, cov1 = [2, 2], [[1, 0.5], [0.5, 1]]
    mean2, cov2 = [-2, -2], [[1, -0.5], [-0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 2)
    data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 2)
    data = np.vstack([data1, data2])
    np.random.shuffle(data)
    return data

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

def encoder(X, W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar):
    hidden = np.tanh(np.dot(X, W1) + b1)
    mu = np.dot(hidden, W2_mu) + b2_mu
    logvar = np.dot(hidden, W2_logvar) + b2_logvar
    return mu, logvar

def reparameterize(mu, logvar):
    std = np.exp(0.5 * logvar)
    epsilon = np.random.randn(*std.shape)
    return mu + epsilon * std

def decoder(z, W3, b3):
    reconstruction = 1 / (1 + np.exp(-np.dot(z, W3) - b3))  # Sigmoid activation
    return reconstruction

def compute_loss(X, reconstruction, mu, logvar):
    reconstruction_loss = -np.sum(X * np.log(reconstruction + 1e-8) + (1 - X) * np.log(1 - reconstruction + 1e-8), axis=1)
    kl_divergence = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar), axis=1)
    return np.mean(reconstruction_loss + kl_divergence)

def train_vae(data, latent_dim=2, learning_rate=0.001, epochs=100):
    input_dim = data.shape[1]
    W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar, W3, b3 = initialize_weights(input_dim, latent_dim)
    losses = []

    for epoch in range(epochs):
        mu, logvar = encoder(data, W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar)
        z = reparameterize(mu, logvar)
        reconstruction = decoder(z, W3, b3)
        loss = compute_loss(data, reconstruction, mu, logvar)
        losses.append(loss)

        # Backpropagation
        reconstruction_grad = reconstruction - data
        dW3 = np.dot(z.T, reconstruction_grad) / data.shape[0]
        db3 = np.mean(reconstruction_grad, axis=0)
        dz = np.dot(reconstruction_grad, W3.T)
        dmu = dz
        dlogvar = dz * (z - mu) * 0.5 * np.exp(-0.5 * logvar)
        dW2_mu = np.dot(mu.T, dmu) / data.shape[0]
        db2_mu = np.mean(dmu, axis=0)
        dW2_logvar = np.dot(logvar.T, dlogvar) / data.shape[0]
        db2_logvar = np.mean(dlogvar, axis=0)
        dhidden = np.dot(dmu, W2_mu.T) + np.dot(dlogvar, W2_logvar.T)
        dW1 = np.dot(data.T, dhidden) / data.shape[0]
        db1 = np.mean(dhidden, axis=0)

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2_mu -= learning_rate * dW2_mu
        b2_mu -= learning_rate * db2_mu
        W2_logvar -= learning_rate * dW2_logvar
        b2_logvar -= learning_rate * db2_logvar
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

    return mu, losses

data = generate_synthetic_data()
latent_dim = 2
mu, losses = train_vae(data, latent_dim=latent_dim)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(mu[:, 0], mu[:, 1], alpha=0.5)
plt.title("Latent Space")
plt.xlabel("z1")
plt.ylabel("z2")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.savefig('output.png')
plt.show()