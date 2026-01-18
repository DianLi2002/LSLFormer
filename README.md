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

Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

Dian Li, Siyuan Hao, Cheng Fang, Yuanxin Ye. LSLFormer: A Lightweight Spectral-LiDAR Fusion Network for Remote Sensing Image Classification, IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026, DOI: 10.1109/TGRS.2026.3654154.

    @ARTICLE{11352989,
      author={Li, Dian and Hao, Siyuan and Fang, Cheng and Ye, Yuanxin},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={LSLFormer: A Lightweight Spectral-LiDAR Fusion Network for Remote Sensing Image Classification}, 
      year={2026},
      volume={},
      number={},
      pages={1-1},
      keywords={Laser radar;Transformers;Feature extraction;Computational modeling;Computational efficiency;Accuracy;Image classification;Geology;Data models;Reviews;Hyperspectral images;LiDAR;data fusion;Transformer;lightweight;remote sensing},
      doi={10.1109/TGRS.2026.3654154}
    }

  @article{11352989,
    author={Li, Dian and Hao, Siyuan and Fang, Cheng and Ye, Yuanxin},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={LSLFormer: A Lightweight Spectral-LiDAR Fusion Network for Remote Sensing Image Classification}, 
    year={2026},
    volume={},
    number={},
    pages={1-1},
    keywords={Laser radar;Transformers;Feature extraction;Computational modeling;Computational efficiency;Accuracy;Image classification;Geology;Data models;Reviews;Hyperspectral images;LiDAR;data fusion;Transformer;lightweight;remote sensing},
    doi={10.1109/TGRS.2026.3654154}
  }
