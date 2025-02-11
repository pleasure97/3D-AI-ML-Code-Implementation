import torch

def L1_Loss(network_output, ground_truth):
    return torch.abs((network_output - ground_truth)).mean()
