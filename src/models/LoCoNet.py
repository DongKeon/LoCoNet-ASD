import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from einops import rearrange

from models.attentionLayer import CrossAttentionModule, DualAttentionModule

import pdb


class LSCM(nn.Module):
    def __init__(self, num_blocks, **kwargs):
        super(LSCM, self).__init__()
        self.blocks = nn.ModuleList([LSCMBlock(**kwargs) 
                                     for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LSCMBlock(nn.Module):
    def __init__(self, s, k, d_model, nhead, dropout):
        super(LSCMBlock, self).__init__()
        self.sim = SIM(s, k, d_model)
        self.lim = LIM(d_model, nhead, dropout)

    def forward(self, x):
        x = self.sim(x)
        x = self.lim(x)
        return x


class SIM(nn.Module):
    def __init__(self, s, k, hidden_dim):
        super(SIM, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, (s, k), padding=((s - 1) // 2, (k - 1) // 2))
        self.ln1 = LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln2 = LayerNorm(hidden_dim)

    def forward(self, u_l):
        u_l_s = self.conv(rearrange(u_l, 'b s t c -> b c s t'))
        u_l_s = self.ln1(rearrange(u_l_s, 'b c s t -> b s t c'))
        u_s_l = self.ln2(self.mlp(u_l_s) + u_l)
        return u_s_l


class LIM(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(LIM, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ln1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ln2 = LayerNorm(d_model)

    def forward(self, u_s_l):
        qkv = rearrange(u_s_l, 'b s t c -> t (b s) c')
        h_l, _ = self.self_attention(qkv, qkv, qkv)
        h_l = self.ln1(h_l + qkv)
        u_l_next = self.ln2(self.mlp(h_l) + h_l)
        return rearrange(u_l_next, 't (b s) c -> b s t c', b=u_s_l.size(0))


class LoCoNet(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, nhead=8, dropout=0.0, num_blocks=2):
        super(LoCoNet, self).__init__()
        self.mlp_a = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.datt = DualAttentionModule(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.lscm = LSCM(num_blocks=num_blocks, s=3, k=7, d_model=hidden_dim*2, 
                         nhead=nhead, dropout=dropout)

    def forward(self, e_a, e_v):
        e_a = self.mlp_a(e_a)
        e_v = self.mlp_v(e_v)
        u = self.datt(e_a, e_v)
        u = self.lscm(u)
        #u = u[:,0, u.size(1) // 2, :] 
        u = u.contiguous().view(u.size(0), -1)

        return u

