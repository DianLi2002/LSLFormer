<div align="center">

# LSLFormer: A Lightweight Spectral-LiDAR Fusion Network
### for Remote Sensing Image Classification

[![IEEE TGRS](https://img.shields.io/badge/IEEE_TGRS-10.1109/TGRS.2026.3654154-blue?logo=ieee)](https://doi.org/10.1109/TGRS.2026.3654154)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b?logo=arxiv)](https://arxiv.org/) <!-- 如有 arXiv 链接请替换 -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dian Li, Siyuan Hao, Cheng Fang, Yuanxin Ye**  
*IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026*

</div>

---

## 📸 Overview

<p align="center">
  <img src="./LSLF.png" width="90%" alt="LSLFormer architecture overview"/>
  <br/>
  <b>An overview illustration of the proposed Lightweight Spectral-LiDAR Fusion Network</b>
</p>

---

## 🧠 Method Highlights

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="./sla.png" width="100%" />
      <br>
      <b>🔬 Detailed structure of Spectral-LiDAR Attention</b>
    </td>
    <td align="center" width="50%">
      <img src="./flops.png" width="100%" />
      <br>
      <b>📊 Computation cost & Overall Accuracy of different methods</b>
    </td>
  </tr>
</table>

---

## 📦 Dataset

The **Houston2013** dataset used in our experiments can be downloaded from:

| Platform | Link | Access |
|----------|------|--------|
| Google Drive | [Download](https://drive.google.com/drive/folders/1Op5O5UhlPWZK6ng9IYT1MzBftsool7jt?usp=drive_link) | Public |
| BaiduYun | [Download](https://pan.baidu.com/s/1T5m8ADyHL0gSkzIh8bp8dg) | Code: `f391` |

---

## 🚀 Usage

### Train the model

```bash
python train.py \
    --patches=7 \
    --band_patches=3 \
    --weight_decay=5e-3 \
    --dataset='houston2013' \
    --flag_test='train'
```bash

🧪 Test the model

python train.py \
    --patches=7 \
    --band_patches=3 \
    --weight_decay=5e-3 \
    --dataset='houston2013' \
    --flag_test='test'


