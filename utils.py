import torch
import numpy as np

def soft_update(target_model, model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def get_obs(observation):
    obs = []
    for o in observation.values():
        if type(o) is np.ndarray:
            obs += list(o)
        else:
            obs.append(o)
    return np.array([obs], dtype=np.float32)

def calc_priority(td_loss, eta=0.9):
    return eta * max(list(td_loss)) + (1. - eta) * (sum(list(td_loss)) / len(td_loss))

def invertical_vf(x):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1)