import torch
from sklearn import metrics
import numpy as np

def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)

def compute_similarity(x, y):
    """Compute the distance between x and y vectors."""
    # similarity is negative distance
    return -euclidean_distance(x, y)
