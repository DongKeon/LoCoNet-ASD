# LoCoNet
LoCoNet: Long-Short Context Network for Active Speaker Detection (2023 CVPR)

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


2) Training / Validation
```
bash run.sh conf/LoCoNet/ResNet18/large [gpu_id] 1
```

## Note
- Code for training only LoCoNet backend with pre-trained feature provided by [SPELL](https://github.com/SRA2/SPELL).
