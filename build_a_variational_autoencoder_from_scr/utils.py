import numpy as np

def generate_gaussian_data(num_samples):
    cluster_centers = [(-5, -5), (5, 5), (-5, 5), (5, -5)]
    data = []
    for center in cluster_centers:
        samples = np.random.normal(loc=center, scale=1.0, size=(num_samples // len(cluster_centers), 2))
        data.append(samples)
    return np.vstack(data)