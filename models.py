import torch
import torch.nn as nn
import numpy as np

init_w = 3e-3
init_b = 3e-4

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# 300-200
class ActorNet(nn.Module):
    def __init__(self, obs_size, n_actions, cuda_id):
        super(ActorNet, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size, out_features=128)
        self.l2 = nn.LSTMCell(128, 128)
        self.l3 = nn.Linear(in_features=128, out_features=n_actions)

        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight_ih.data = fanin_init(self.l2.weight_ih.data.size())
        self.l2.weight_hh.data = fanin_init(self.l2.weight_hh.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.fill_(init_b)

        self.hx = None
        self.cx = None

        self.cuda_id = cuda_id

    def __call__(self, x):
        x = torch.tanh(self.l1(x))
        if self.hx is None: # 200
            self.hx = torch.zeros((x.size()[0] ,128)).cuda(self.cuda_id)
            self.cx = torch.zeros((x.size()[0] ,128)).cuda(self.cuda_id)
        self.hx, self.cx = self.l2(x, (self.hx, self.cx))
        x = torch.tanh(self.hx)
        x = torch.tanh(self.l3(x))
        return x

    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx
    
    def reset_state(self):
        self.hx = None
        self.cx = None

    def get_state(self):
        if self.hx is None:
            return np.zeros((1, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)
        else:
            return self.hx.clone().detach().cpu().numpy(), self.cx.clone().detach().cpu().numpy()

class CriticNet(nn.Module): # 400-300
    def __init__(self, obs_size, n_actions, cuda_id):
        super(CriticNet, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size + n_actions, out_features=128)
        self.l2 = nn.LSTMCell(128, 128)
        self.l3 = nn.Linear(in_features=128, out_features=n_actions)

        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight_ih.data = fanin_init(self.l2.weight_ih.data.size())
        self.l2.weight_hh.data = fanin_init(self.l2.weight_hh.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.fill_(init_b)

        self.hx = None
        self.cx = None

        self.cuda_id = cuda_id

    def __call__(self, x, a):
        x = torch.cat((x,a), 1)
        x = torch.tanh(self.l1(x))
        if self.hx is None: # 300
            self.hx = torch.zeros((x.size()[0] ,128)).cuda(self.cuda_id)
            self.cx = torch.zeros((x.size()[0] ,128)).cuda(self.cuda_id)
        self.hx, self.cx = self.l2(x, (self.hx, self.cx))
        x = torch.tanh(self.hx)
        x = self.l3(self.hx)
        return x

    def reset_state(self):
        self.hx = None
        self.cx = None

    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx

    def get_state(self):
        if self.hx is None:
            return np.zeros((1, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)
        else:
            return self.hx.clone().detach().cpu().numpy(), self.cx.clone().detach().cpu().numpy()