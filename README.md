# LSLFormer: A Lightweight Spectral-LiDAR Fusion Network for Remote Sensing Image Classification

Dian Li, Siyuan Hao, Cheng Fang, Yuanxin Ye

___________
![alt text](./LSLF.png)

# Dataset

You can download the Houston2013 dataset we use from the following links of google drive or baiduyun:

Google drive: https://drive.google.com/drive/folders/1Op5O5UhlPWZK6ng9IYT1MzBftsool7jt?usp=drive_link

Baiduyun: https://pan.baidu.com/s/1T5m8ADyHL0gSkzIh8bp8dg?pwd=f391 (access code: f391)

# Train
python train.py --patches=7 --band_patches=3 --weight_decay=5e-3 --dataset='houston2013' --flag_test='train'
# Test
python train.py --patches=7 --band_patches=3 --weight_decay=5e-3 --dataset='houston2013' --flag_test='test'
