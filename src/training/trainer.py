import sys
import time
import subprocess
import tqdm
import pandas
import torch
import torch.nn as nn
from einops import rearrange

# models
from models.ASCNet      import ASCNet
from models.TalkNet     import TalkNet
from models.LoCoNet     import LoCoNet

from loss               import lossAV, lossA, lossV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import pdb


class TrainModule(nn.Module):
    def __init__(self, args, writer):
        super(TrainModule, self).__init__()        
        # Model{{{
        self.model_type = args.model_type
        self.feat_dim = args.feat_dim
        if self.model_type == 'TalkNet':
            self.model = TalkNet(in_dim=args.feat_dim, d_model=args.hidden_dim, nhead=8)
            self.final_dim = args.feat_dim * 2

        elif self.model_type == "ASCNet":
            self.model = ASCNet(in_dim=args.feat_dim,
                                 hidden_dim=args.hidden_dim)
            self.final_dim = args.hidden_dim*args.time_length*args.candidate_speakers

        elif self.model_type == "LoCoNet":
            self.model = LoCoNet(in_dim=args.feat_dim, hidden_dim=args.hidden_dim, 
                                 dropout=args.dropout, num_blocks=args.num_blocks)
            self.final_dim = args.hidden_dim*2*args.time_length*args.candidate_speakers

        else:
            raise NotImplementedError("Model not implemented")
        self.model = self.model.cuda()
        # }}}

        # Loss{{{
        self.aux_loss = args.aux_loss
        self.lossAV = lossAV(self.final_dim).cuda()
        if args.aux_loss:
            self.lossA = lossA().cuda()
            self.lossV = lossV().cuda()# }}}

        # Optimizer{{{
        if args.optimizer == "Adam":
            self.optim = torch.optim.Adam(self.parameters(), lr=args.lr)
        elif args.optimizer == "SGD":
            self.optim = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.9, 
                                        weight_decay=args.lrDecay)
        else:
            raise NotImplementedError("Optimizer not implemented")# }}}

        # Scheduler{{{
        if args.scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 
                                                             step_size=args.step_size, gamma=args.lrDecay)
        elif args.scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, 
                                                                milestones=[10, 20, 30, 40, 50],
                                                                gamma=args.lrDecay)
        else:
            raise NotImplementedError("Scheduler not implemented")# }}}

        self.writer= writer
        self.mAP_log = False

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        if self.mAP_log:
            pred_lst = []
            label_lst = []
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (feat_data, labels) in enumerate(loader):
            feat_data = feat_data.cuda()
            self.optim.zero_grad()
            if self.model_type == 'TalkNet':# {{{
                feat_data = feat_data[:,:,:,0]
                feat_data = torch.swapaxes(feat_data, 1, 2)
                audioEmbed = feat_data[:,:,:self.feat_dim]
                visualEmbed = feat_data[:,:,self.feat_dim:]
                # Cross Attention
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                # Backend (Decoder)
                outsAV                  = self.model.forward_av_backend(audioEmbed, visualEmbed)  
                if self.aux_loss:
                    outsA                   = self.model.forward_audio_backend(audioEmbed)
                    outsV                   = self.model.forward_visual_backend(visualEmbed)
                #}}}
            elif self.model_type == 'ASCNet':# {{{
                outsAV = self.model(feat_data)# }}}
            elif self.model_type == 'LoCoNet':# {{{
                feat_data = rearrange(feat_data, 'b c t s -> b s t c')
                audioEmbed = feat_data[:, :, :, :self.feat_dim]
                visualEmbed = feat_data[:, :, :, self.feat_dim:]
                outsAV = self.model(audioEmbed, visualEmbed)
                # }}}

            # Loss{{{
            labels                  = labels.cuda() 
            nlossAV, _, predLabel, prec     = self.lossAV.forward(outsAV, labels)
            if self.aux_loss:
                nlossA                  = self.lossA.forward(outsA, labels)
                nlossV                  = self.lossV.forward(outsV, labels)
                nloss                    = nlossAV + 0.4 * nlossA + 0.4 * nlossV 
            else:
                nloss                    = nlossAV
            loss += nloss.detach().cpu().numpy() 
            top1 += prec 
            if self.mAP_log:
                label_lst.extend(labels.cpu().numpy().tolist())
                pred_lst.extend(predLabel.cpu().numpy().tolist())
                epoch_auc = roc_auc_score(label_lst, pred_lst) 
            # }}}

            # Optimizer{{{
            nloss.backward()
            self.optim.step() # }}}

            # Logging{{{
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

            global_step = (epoch - 1) * len(loader) + num
            # TensorBoard logging
            self.writer.add_scalar('Measure/Train/Loss', nloss.item(), global_step)
            self.writer.add_scalar('Measure/Train/Acc', 100 * (prec/len(labels)), global_step)
            self.writer.add_scalar('Epoch', epoch, global_step)
# }}}

        sys.stdout.write("\n")      
        return loss/num, lr


    def validate_network(self, epoch, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        pred_lst = []
        label_lst = []
        val_loss, top1, index = 0, 0, 0
        for num, (feat_data, labels) in enumerate(loader):
            with torch.no_grad():                
                feat_data = feat_data.cuda()
                if self.model_type == 'TalkNet':# {{{
                    feat_data = feat_data[:,:,:,0]
                    feat_data = torch.swapaxes(feat_data, 1, 2)
                    audioEmbed = feat_data[:,:,:self.feat_dim]
                    visualEmbed = feat_data[:,:,self.feat_dim:]
                    # Cross Attention
                    audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                    # Backend (Decoder)
                    outsAV                  = self.model.forward_av_backend(audioEmbed, visualEmbed)  
                    if self.aux_loss:
                        outsA                   = self.model.forward_audio_backend(audioEmbed)
                        outsV                   = self.model.forward_visual_backend(visualEmbed)
                    #}}}
                elif self.model_type == 'ASCNet':# {{{
                    outsAV = self.model(feat_data)# }}}
                elif self.model_type == 'LoCoNet':# {{{
                    feat_data = rearrange(feat_data, 'b c t s -> b s t c')
                    audioEmbed = feat_data[:, :, :, :self.feat_dim]
                    visualEmbed = feat_data[:, :, :, self.feat_dim:]
                    outsAV = self.model(audioEmbed, visualEmbed)
                    #}}}

                # Loss{{{
                labels = labels.cuda()             
                nlossAV, predScore, predLabel, prec = self.lossAV.forward(outsAV, labels)    
                label_lst.extend(labels.cpu().numpy().tolist())
                pred_lst.extend(predScore.cpu().numpy()[:, 1].tolist()) # }}}

            top1 += prec
            val_loss += nlossAV
            index += len(labels)

        epoch_auc   = roc_auc_score(label_lst, pred_lst) * 100
        epoch_mAP   = average_precision_score(label_lst, pred_lst) * 100
        epoch_acc   = top1 / index * 100
        val_loss    = val_loss / index * 100
        self.writer.add_scalar('Measure/Val/mAP', epoch_mAP, epoch)
        self.writer.add_scalar('Measure/Val/AUC', epoch_auc, epoch)
        self.writer.add_scalar('Measure/Val/Acc', epoch_acc, epoch)
        self.writer.add_scalar('Measure/Val/Loss', val_loss, epoch)
        return epoch_mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
