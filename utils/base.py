import torch

def get_item(t):
    return t.clone().detach().item()