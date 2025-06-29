import torch

def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)

def pairwise_loss(x, y, labels, margin=1.0):
    loss = torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    return loss, euclidean_distance(x, y)
