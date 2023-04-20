#!/bin/bash
export PYTHONPATH=`pwd`/src:$PYTHONPATH
export PATH=/root/miniconda3/envs/LoCoNet/bin:$PATH

conf_path=$1
exp_name=${conf_path#conf/}
exp_dir=exp/${exp_name}

gpu_id=$2
stage=$3

# Config
train_conf=$conf_path/train.yaml
infer_conf=$conf_path/infer.yaml


n_cpu=64

if [ $stage -le 1 ]; then 
  CUDA_VISIBLE_DEVICES=$gpu_id \
  python main.py -c $train_conf \
                 --AVA_data_path dataset \
                 --save_path $exp_dir \
                 --num_workers $n_cpu
  exit
fi


if [ $stage -le 2 ]; then 
  python main.py --AVA_data_path dataset --save_path $save_path --inference
  exit
fi

