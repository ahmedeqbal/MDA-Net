# MDA-Net

This repository contains the original implementation of "**[MDA-Net: Multiscale dual attention-based network for breast lesion segmentation using ultrasound images](https://www.sciencedirect.com/science/article/pii/S1319157821002895)**" in PyTorch library. This paper has been published in "*Journal of King Saud University - Computer and Information Sciences*"

## Proposed Architecture:
<img src="paper_images/Proposed%20_architecture.jpg">

## Dual attention block:
<img src="paper_images/Dual_attention_block.jpg">

## Results on Private Ultrasound dataset:
<img src="paper_images/BUS_segmentation.jpg">

## Results on RIDER breast MRI dataset:
<img src="paper_images/MRI_segmentation.jpg">

## Pretrained weights:

Download from [**Google Drive**](https://drive.google.com/file/d/1x73MYu1fYgEA0-Bu2leNbDtI-t_SKFJr)

## Demo:

A demo can be found in [**Here**](https://github.com/ahmedeqbal/MDA-Net/blob/main/MDA-Net_implementation.ipynb)

## Requirements:

- Python 3.7.9
- PyTorch: 1.6.0
- Numpy: 1.19.2
- OpenCV: 4.5.1
- Scikit-Learn: 0.23.2
- Pandas: 1.1.4
- Matplotlib: 3.3.2

## Cite:

If you use MDA-Net in your project, please cite the following paper:
```
Iqbal, A., & Sharif, M. (2022). "MDA-Net: Multiscale dual attention-based network for breast lesion segmentation using ultrasound images",
Journal of King Saud University - Computer and Information Sciences, 2022. https://doi.org/10.1016/j.jksuci.2021.10.002
```
