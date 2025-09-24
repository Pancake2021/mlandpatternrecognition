"""
Implementation of multivariate normal distribution operations.
"""
import numpy as np
from typing import Tuple, List


def generate_multivariate_normal(mean: np.ndarray, cov: np.ndarray, n_samples: int = 1) -> np.ndarray:
    """
    Generate samples from a multivariate normal distribution using Cholesky decomposition.
    
    Args:
        mean: Mean vector of the distribution (n_features,)
        cov: Covariance matrix (n_features, n_features)
        n_samples: Number of samples to generate
        
    Returns:
        Array of shape (n_samples, n_features)
    """
    n_features = len(mean)
    
    # Generate standard normal samples
    z = np.random.standard_normal((n_samples, n_features))
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(cov)
    
    # Transform to desired distribution
    samples = mean + np.dot(z, L.T)
    
    return samples


def estimate_parameters(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean and covariance matrix from samples.
    
    Args:
        samples: Array of shape (n_samples, n_features)
        
    Returns:
        Tuple of (mean, covariance_matrix)
    """
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    cov = np.dot(centered.T, centered) / (len(samples) - 1)
    return mean, cov


def mahalanobis_distance(x: np.ndarray, y: np.ndarray, cov: np.ndarray) -> float:
    """
    Calculate Mahalanobis distance between two points.
    
    Args:
        x: First point (n_features,)
        y: Second point (n_features,)
        cov: Covariance matrix (n_features, n_features)
        
    Returns:
        Mahalanobis distance
    """
    diff = x - y
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))


def bhattacharyya_distance(
    mean1: np.ndarray, 
    cov1: np.ndarray, 
    mean2: np.ndarray, 
    cov2: np.ndarray
) -> float:
    """
    Calculate Bhattacharyya distance between two multivariate normal distributions.
    
    Args:
        mean1: Mean of first distribution (n_features,)
        cov1: Covariance of first distribution (n_features, n_features)
        mean2: Mean of second distribution (n_features,)
        cov2: Covariance of second distribution (n_features, n_features)
        
    Returns:
        Bhattacharyya distance
    """
    # Average covariance
    cov_avg = (cov1 + cov2) / 2
    diff = mean1 - mean2
    
    # Calculate the terms
    term1 = 0.125 * np.dot(np.dot(diff.T, np.linalg.inv(cov_avg)), diff)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    det_avg = np.linalg.det(cov_avg)
    term2 = 0.5 * np.log(det_avg / np.sqrt(det_cov1 * det_cov2))
    
    return term1 + term2
