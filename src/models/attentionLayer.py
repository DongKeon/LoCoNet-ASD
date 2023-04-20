import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

from einops import rearrange

import pdb


class CrossAttentionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(CrossAttentionModule, self).__init__()
        self.MHA = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.MLP = nn.Linear(d_model, d_model)

    def forward(self, query, key_value):
        q = rearrange(query, 'b s t c -> t (b s) c')
        kv = rearrange(key_value, 'b s t c -> t (b s) c')
        attn_output, _ = self.MHA(q, kv, kv)
        output = kv + self.dropout(attn_output)
        output = self.MLP(output)
        return rearrange(output, 't (b s) c -> b s t c', b=query.size(0))


class DualAttentionModule(nn.Module):
    """ Audio-Visual Attention Block (Dual Attention) """
    def __init__(self, d_model, nhead, dropout=0.0):
        super(DualAttentionModule, self).__init__()
        self.crossA2V = CrossAttentionModule(d_model=d_model, nhead=nhead)
        self.crossV2A = CrossAttentionModule(d_model=d_model, nhead=nhead)

    def forward(self, e_a, e_v):
        h_v = self.crossA2V(query=e_v, key_value=e_a)
        h_a = self.crossV2A(query=e_a, key_value=e_v)
        u = torch.cat((h_v, h_a), 3)    

        return u


"""From TalkNet"""
class AttentionLayer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(in_dim, nhead, dropout=dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(in_dim)

        self.linear1 = nn.Linear(in_dim, d_model)
        self.activation = F.relu

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, in_dim)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(in_dim)


    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src


"""From ASC"""
class SelfAttentionModule(nn.Module):# {{{
    def __init__(self, inplanes, planes, stride=1):
        super(SelfAttentionModule, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.theta = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.theta_bn = norm_layer(planes)

        self.phi = nn.Conv2d(inplanes, planes, kernel_size=1,
                             stride=1, padding=0, bias=True)
        self.phi_bn = norm_layer(planes)

        self.gamma = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.gamma_bn = norm_layer(planes)

        self.omega = nn.Conv2d(planes, inplanes, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.omega_bn = norm_layer(inplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_size = x.size()
        t = self.theta(x)
        t = self.theta_bn(t)

        p = self.phi(x)
        p = self.phi_bn(p)

        g = self.gamma(x)
        g = self.gamma_bn(g)

        t = t.reshape(t.size(0), t.size(1), t.size(2)*t.size(3))
        t = t.permute(0, 2, 1)
        p = p.reshape(p.size(0), p.size(1), p.size(2)*p.size(3))
        attention = torch.matmul(t, p)
        att_size = attention.size()
        attention = nn.functional.softmax(attention.view(att_size[0], -1), dim=1)
        attention = attention.reshape(att_size)

        g = g.reshape(g.size(0), g.size(1), g.size(2)*g.size(3))
        g = g.permute(0, 2, 1)

        sat = torch.matmul(attention, g)
        sat = sat.permute(0, 2, 1)
        sat = sat.reshape(g.size(0), g.size(2), x_size[2], x_size[3])

        sat = self.omega(sat)
        sat = self.omega_bn(sat)

        return x + sat# }}}
