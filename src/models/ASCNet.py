import torch
import torch.nn as nn

from models.attentionLayer    import SelfAttentionModule


class ASCNet(nn.Module):
    def __init__(self, clip_number=11, candidate_speakers=3,
                 in_dim=512, hidden_dim=128):
        super(ASCNet, self).__init__()
        self.sat = SelfAttentionModule(in_dim*2, hidden_dim)
        self.lstm = nn.LSTM(in_dim*2, hidden_dim, batch_first=True)

    def forward(self, x):
        # Pairwise Attention
        x = self.sat(x)

        # Temporal Module
        x = x.reshape(x.size(0),x.size(1),-1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        # Final Prediction
        x = x.contiguous().view(x.size(0), -1)
        return x

