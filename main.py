#!/usr/bin/env python3

# Copyright 2023 AiTeR, GIST (Author: Dongkeon Park)
# Licensed under the MIT license.

import os
import argparse
import warnings
import glob
import yamlargparse

import torch
import random
import numpy as np

from training.trainer import TrainModule

from utils.tools import *
from utils.logging_util import *
from data.dataset import ASCFeaturesDataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pdb

def parse_arguments(): #{{{
    parser = yamlargparse.ArgumentParser(description='ASD training')
    parser.add_argument('-c', '--config', help='config file path', action=yamlargparse.ActionConfigFile)
    # Model{{{
    parser.add_argument('--model_type',         type=str,   default="talkNet",
                        help='Model name')
    parser.add_argument('--feat_dim',           type=int,   default=512,
                        help='Feature dimension')
    parser.add_argument('--hidden_dim',         type=int,   default=512,
                        help='Hidden dimension')
    parser.add_argument('--nhead',              type=int,   default=8,
                        help='Number of heads')
    parser.add_argument('--num_blocks',         type=int,   default=2,
                        help='Number of blocks')
    parser.add_argument('--dropout',            type=float, default=0.0,
                        help='Dropout rate')
    # }}}
    # Loss{{{
    parser.add_argument('--aux_loss',           dest='aux_loss', action='store_true', help='Use auxiliary loss')# }}}
    # Structured Context Ensemble{{{
    parser.add_argument('--time_length',        type=int,   default=11,
                        help='Time length of input')
    parser.add_argument('--time_stride',        type=int,   default=4,
                        help='Time stride of input')
    parser.add_argument('--candidate_speakers', type=int,   default=3,
                        help='Number of candidate speakers') # }}}
    # Optimizer{{{
    parser.add_argument('--lr',                 type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--lrDecay',            type=float, default=0.95,
                        help='Learning rate decay rate')
    parser.add_argument('--step_size',          type=int,   default=1,
                        help='Learning rate decay step size')
    parser.add_argument('--optimizer',          type=str,   default="Adam",
                        help='Optimizer')
    parser.add_argument('--scheduler',          type=str,   default="StepLR",
                        help='Scheduler')# }}}
    # Training details{{{
    parser.add_argument('--maxEpoch',           type=int,   default=5,
                        help='Maximum number of epochs')
    parser.add_argument('--testInterval',       type=int,   default=1,
                        help='Test and save every [testInterval] epochs')
    parser.add_argument('--batch_size',         type=int,   default=64,
                        help='Batch size')
    parser.add_argument('--num_workers',        type=int,   default=4,
                        help='Number of loader threads')
    parser.add_argument('--seed',               type=int,   default=1,
                        help='Random seed')# }}}
    # ETC{{{
    ## Mode 
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    parser.add_argument('--inference',      dest='inference', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    ## Data path
    parser.add_argument('--AVA_data_path',  type=str, default="/data08/AVA", help='Save path of AVA dataset')
    parser.add_argument('--save_path',     type=str, default="exps/exp1")
    parser.add_argument('--feat_path',     type=str, default="features")
    ## Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # }}}

    args = parser.parse_args()
    #args.save_path = args.save_path + \
    #                '_len_' + str(args.time_length) + \
    #                '_stride_'+str(args.time_stride) + \
    #                '_speakers_'+str(args.candidate_speakers)
    print(f"Save path is {args.save_path}")
    args = init_args(args)

    return args

    # }}}


def set_seed(args):# {{{
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True# }}}


def create_data_loaders(args):# {{{
    """TalkNet-Based"""
    """
    # {{{
    from data.dataLoader import train_loader, val_loader
    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                          **vars(args))
    train_loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=True, num_workers=args.num_workers)

    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        **vars(args))
    val_loader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle=False, num_workers=args.num_workers)
    # }}}
    """
    """ASC-Based"""
    # {{{
    dataset_train = ASCFeaturesDataset(os.path.join(args.feat_path, 'train_forward/*.csv'),
                                       time_length=args.time_length,
                                       time_stride=args.time_stride,
                                       candidate_speakers=args.candidate_speakers,
                                       feat_dim=args.feat_dim)
    dataset_val = ASCFeaturesDataset(os.path.join(args.feat_path, 'val_forward/*.csv'),
                                       time_length=args.time_length,
                                       time_stride=args.time_stride,
                                       candidate_speakers=args.candidate_speakers,
                                       feat_dim=args.feat_dim)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # }}}
    return train_loader, val_loader# }}}


def initialize(args, writer):# {{{
    model_file = max(glob.glob(f"{args.modelSavePath}/model_0*.model"), default=None)
    epoch = int(os.path.splitext(os.path.basename(model_file))[0][6:]) + 1 if model_file else 1
    trainer = TrainModule(args, writer=writer)
    trainer.loadParameters(model_file) if model_file else None

    return trainer, epoch# }}}


def train(trainer, train_loader, val_loader, args):# {{{
    mAPs = []
    score_file = open(args.scoreSavePath, "a+")
    log_model(score_file, trainer.model)

    #validate(trainer, 0, args, val_loader, mAPs, score_file, loss=0, lr=0)
    for epoch in range(args.startEpoch, args.maxEpoch):
        loss, lr = trainer.train_network(epoch=epoch, loader=train_loader, **vars(args))

        if epoch % args.testInterval == 0:        
            validate(trainer, epoch, args, val_loader, mAPs, score_file, loss, lr)

    score_file.close()# }}}


def validate(trainer, epoch, args, val_loader, mAPs, score_file, loss, lr):# {{{
    model_path = f"{args.modelSavePath}/model_{epoch:04d}.model"
    trainer.saveParameters(model_path)
    current_mAP = trainer.validate_network(epoch=epoch, loader=val_loader, **vars(args))
    mAPs.append(current_mAP)
    best_mAP = max(mAPs)
    print_evaluation_results(epoch, current_mAP, best_mAP)
    log_evaluation_results(score_file, epoch, lr, loss, current_mAP, best_mAP)
    # }}}



def main():
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    set_seed(args)
    train_loader, val_loader = create_data_loaders(args)
    writer = SummaryWriter(log_dir=args.save_path)
    trainer, start_epoch = initialize(args, writer)
    args.startEpoch = start_epoch
    train(trainer, train_loader, val_loader, args)

if __name__ == '__main__':
    main()
