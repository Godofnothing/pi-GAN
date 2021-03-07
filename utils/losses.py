import torch.nn.functional as F
from torch import nn

def get_GAN_losses(mode='default'):
    if mode == 'default':
        return lambda logits, labels : F.relu(1 + (2 * labels - 1) * logits).mean(), \
               lambda logits, labels : logits.mean()
    elif mode == 'log':
        return nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError("Unknown loss")