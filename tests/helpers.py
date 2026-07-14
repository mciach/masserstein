import numpy as np


def gaussian_profile(centers, heights, sd=0.02, step=0.002, margin=0.2):
    """Build a profile-mode confs list: a sum of Gaussians sampled on a regular grid."""
    centers = np.asarray(centers, dtype=float)
    heights = np.asarray(heights, dtype=float)
    mz = np.arange(centers.min() - margin, centers.max() + margin, step)
    intensity = np.zeros_like(mz)
    for c, h in zip(centers, heights):
        intensity += h * np.exp(-((mz - c) ** 2) / (2 * sd**2))
    return list(zip(mz, intensity))
