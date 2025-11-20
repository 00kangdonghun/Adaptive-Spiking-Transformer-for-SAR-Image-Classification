# SAR-Spikingformer: Adaptive Spiking Transformer for SAR Image Classification, [AAAI 2026 Workshop on Neuromorphic Intelligence](https://openreview.net/pdf?id=98d4g2w0qc)
Based on Spikingformer, this study proposes a lightweight spiking transformer **SAR-Spikingformer** optimized for the classification of synthetic aperture radar (SAR) images.

SAR-Spikingformer introduces SAR-specific data augmentation and a dynamic 'block-halting' mechanism to significantly reduce energy consumption, the key strength of SNN, by up to **27.76%** and **46.53%**, respectively, while maintaining classification accuracy on MSTAR and EuroSAT datasets.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/64926ecf-a90f-46ad-9d5f-2288e0e216fa" />


## Datasets
data prepare: data with the following folder structure, you can extract image정)
## Main results on ImageNet-1K

| Model               | Resolution| T |  Param.     | FLOPs   |  Power |Top-1 Acc| Download |
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |:---: |
| Spikingformer-8-384 | 224x224   | 4 |  16.81M     | 3.88G   | 4.69 mJ   |72.45  |   -    |
| Spikingformer-8-512 | 224x224   | 4 |  29.68M     | 6.52G  | 7.46 mJ   |74.79  |     -  |
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 12.54G  | 13.68 mJ  |75.85  |   [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ) |

All download passwords: abcd

<!-- 
| Spikformer-8-384 | 224x224    |  16.81M     | 6.82G   | 12.43  mJ              |70.24  |
| Spikformer-8-512 | 224x224    |  29.68M     | 11.09G  | 18.82  mJ             |73.38  |
| Spikformer-8-768 | 224x224    |  66.34M     | 22.09G  | 32.07  mJ             |74.81  |
-->

## Main results on CIFAR10/CIFAR100

| Model                | T      |  Param.     | CIFAR10 Top-1 Acc| Download  |CIFAR100 Top-1 Acc|
| :---:                | :---:  | :---:       |  :---:  |:---:   |:---: |
| Spikingformer-4-256  | 4      |  4.15M     | 94.77   |   -   |77.43  |
| Spikingformer-2-384  | 4      |  5.76M     | 95.22   |   -   |78.34  |
| Spikingformer-4-384  | 4      |  9.32M     | 95.61    |   -  |79.09  |
| Spikingformer-4-384-400E  | 4      |  9.32M     | 95.81    | [here](https://pan.baidu.com/s/1mjpD2gtz5ZX0M8N3jobjzA ) |79.21  |

All download passwords: abcd

## Main results on CIFAR10-DVS/DVS128

| Model               | T      |  Param.     |  CIFAR10 DVS Top-1 Acc  | DVS 128 Top-1 Acc|
| :---:               | :---:  | :---:       | :---:                   |:---:            |
| Spikingformer-2-256 | 10     |  2.57M      | 79.9                    | 96.2            |
| Spikingformer-2-256 | 16     |  2.57M      | 81.3                    | 98.3            |



## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2022spikformer,
  title={Spikformer: When spiking neural network meets transformer},
  author={Zhou, Zhaokun and Zhu, Yuesheng and He, Chao and Wang, Yaowei and Yan, Shuicheng and Tian, Yonghong and Yuan, Li},
  journal={arXiv preprint arXiv:2209.15425},
  year={2022}
}
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023},
  url={https://arxiv.org/abs/2304.11954}
}
```
