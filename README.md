# DATR
Dynamic Anchor Boxes and Temporal-Relational Transformers for Enhanced Active Speaker Detection (MMSP 2023)

## Overview
Our happiness circuit on

## Dependency
We used python=3.6, pytorch=1.9.1 in our experiments.

## Code Usage
1) Download the audio-visual features and the annotation csv files from [Google Drive](https://drive.google.com/drive/folders/1fYALbElvIKjqeS8uGTHSeqtOhA6FXuRi?usp=sharing). The directories should look like as follows:
```
|-- features
    |-- resnet18-tsm-aug
        |-- train_forward
        |-- val_forward
    |-- resnet50-tsm-aug
        |-- train_forward
        |-- val_forward
|-- csv_files
    |-- ava_activespeaker_train.csv
    |-- ava_activespeaker_val.csv
```


```
2) bash run.sh conf/LoCoNet/ResNet18/large
```

## Note
- We used the official code of [Active Speakers in Context (ASC)](https://github.com/fuankarion/active-speakers-context) to extract the audio-visual features (Stage-1). Specifically, we used `STE_train.py` and `STE_forward.py` of the ASC repository to train our two-stream ResNet-TSM encoders and extract the audio-visual features. We did not use any other components such as the postprocessing module or the context refinement modules. Please refer to `models_stage1_tsm.py` and the checkpoints from this [link](https://drive.google.com/drive/folders/1oom1XLVv8yAR8TVEmepsQ9Yp0E0SIAXM?usp=sharing) to see how we implanted the TSM into the two-stream ResNets.

