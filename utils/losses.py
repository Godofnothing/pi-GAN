import torch
import torch.nn.functional as F

def log_exp_loss(x):
    """
    non-saturating GAN loss
    """
    return torch.log(1 + torch.exp(-x))

def discriminator_loss(fake_out, real_out, mode = "relu"):
    if mode == 'default':
        return  (F.relu(1+real_out) + F.relu(1-fake_out)).mean()
    elif mode == 'relu':
        return  (F.relu(-real_out) + F.relu(fake_out)).mean()
    elif mode == 'log_exp':
        return (log_exp_loss(-real_out) + log_exp_loss(fake_out)).mean()
    else:
        raise NotImplementedError("Unknown loss")

def generator_loss(fake_out, mode = "relu"):
    if mode == 'default':
        return fake_out.mean()
    elif mode == 'relu':
        return F.relu(fake_out).mean()
    elif mode == 'log_exp':
        return log_exp_loss(fake_out).mean()
    else:
        raise NotImplementedError("Unknown loss")