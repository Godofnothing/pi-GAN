import torch
from torch.autograd import grad as torch_grad

def gradient_penalty(images, output, weight = 10):
    batch_size, device = images.shape[0], images.device
    gradients = torch_grad(
        outputs=output, 
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=device),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    l2 = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
    return weight * l2