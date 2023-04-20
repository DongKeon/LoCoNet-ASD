import torch
import torch.nn as nn

from models.attentionLayer    import AttentionLayer

import pdb


class TalkNet(nn.Module):# {{{
    def __init__(self, in_dim, d_model, nhead=8):
        super(TalkNet, self).__init__()
        self.out_dim = in_dim * 2
        # Audio-visual Cross Attention
        self.crossA2V = AttentionLayer(in_dim=in_dim, d_model=d_model, nhead=nhead)
        self.crossV2A = AttentionLayer(in_dim=in_dim, d_model=d_model, nhead=nhead)

        # Audio-visual Self Attention
        self.selfAV = AttentionLayer(in_dim=in_dim*2, d_model=d_model*2, nhead=nhead)

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src=x1, tar=x2)
        x2_c = self.crossV2A(src=x2, tar=x1)        
        return x1_c, x2_c

    def forward_av_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)    
        x = self.selfAV(src = x, tar = x)       
        #pdb.set_trace()
        x = x[:, x.size(1) // 2, :]
        x = torch.reshape(x, (-1, self.out_dim))
        return x    

    # }}}

