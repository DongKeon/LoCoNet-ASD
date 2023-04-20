import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class lossAV(nn.Module):
    def __init__(self, hidden_units=256):
        super(lossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC        = nn.Linear(hidden_units, 2)
        
    def forward(self, x, labels=None):    
        x = x.squeeze(1)
        x = self.FC(x)
        if labels == None:
            predScore = x[:,1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            nloss = self.criterion(x, labels)
            predScore = F.softmax(x, dim = -1)
            predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum

class lossA(nn.Module):
    def __init__(self):
        super(lossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC        = nn.Linear(128, 2)

    def forward(self, x, labels):    
        x = x.squeeze(1)
        x = self.FC(x)    
        nloss = self.criterion(x, labels)
        return nloss

class lossV(nn.Module):
    def __init__(self):
        super(lossV, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.FC        = nn.Linear(128, 2)

    def forward(self, x, labels):    
        x = x.squeeze(1)
        x = self.FC(x)
        nloss = self.criterion(x, labels)
        return nloss


class lossAV_fused(nn.Module):
    def __init__(self):
        super(lossAV_fused, self).__init__()
        self.criterion  = nn.CrossEntropyLoss()
        self.FC1        = nn.Linear(128, 2)
        self.FC2        = nn.Linear(128, 2)

    def forward(self, x1, x2, labels=None):    
        x1 = x1.squeeze(1)
        x1 = self.FC1(x1)

        x2 = x2.squeeze(1)
        x2 = self.FC2(x2)

        if labels == None:
            predScore = x1[:,1] * 0.5 + x2[:,1] * 0.5
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            nloss1 = self.criterion(x1, labels)
            nloss2 = self.criterion(x2, labels)
            # Option
            nloss = nloss1 + nloss2
            #nloss = nloss1 * 0.5 + nloss2 * 0.5

            predScore1 = F.softmax(x1, dim = -1)
            predScore2 = F.softmax(x2, dim = -1)
            predScore = predScore1 * 0.5 + predScore2 * 0.5

            predLabel = torch.round(predScore)[:,1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum


        nloss = self.criterion(x, labels)
