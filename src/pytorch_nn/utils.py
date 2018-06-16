import numpy as np
import torch
import torch.nn as nn


def init_weights(config):
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight.data, 0)
            nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain(config['activation']))
            m.bias.data.zero_()
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        if type(m) == nn.ConvTranspose2d:
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
    fc = _init_weights
    return fc

def add_gradient_noise(parameters, step):
    r"""Add gradient noise from normal distribution

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        clip_value (float or int): maximum allowed value of the gradients
            The gradients are clipped in the range [-clip_value, clip_value]
    """

    eta = 0.01
    gamma = 0.55
    variance = eta / ((1 + float(step))**gamma)
    std = np.sqrt(variance)


    for p in filter(lambda p: p.grad is not None, parameters):
        noise = torch.normal(torch.zeros_like(p.grad), std)
        p.grad.data.add_(noise)
