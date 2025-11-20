# SAR-Spikingformer: Adaptive Spiking Transformer for SAR Image Classification, [AAAI 2026 Workshop on Neuromorphic Intelligence](https://openreview.net/pdf?id=98d4g2w0qc)
Based on Spikingformer, this study proposes a lightweight spiking transformer **SAR-Spikingformer** optimized for the classification of synthetic aperture radar (SAR) images.

SAR-Spikingformer introduces SAR-specific data augmentation and a dynamic 'block-halting' mechanism to significantly reduce energy consumption, the key strength of SNN, by up to **27.76%** and **46.53%**, respectively, while maintaining classification accuracy on MSTAR and EuroSAT datasets.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/64926ecf-a90f-46ad-9d5f-2288e0e216fa" />


## Datasets
data prepare: data with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
(수정)
### MSTAR-10classes
https://www.kaggle.com/datasets/ravenchencn/mstar-10-classes
```
│MSTAR-10classes/
├──train/
│  ├── 2S1
│  ├── BMP2
│  ├── BRDM2
│  ├── BTR60
│  ├── ......
│
├──test/
│  ├── 2S1
│  ├── BMP2
│  ├── BRDM2
│  ├── BTR60
│  ├── ......
```
### EuroSAT
https://zenodo.org/records/7711810#.ZAm3k-zMKEA
```
│EuroSAT/
├──train/
│  ├── AnnualCrop
│  ├── Forest
│  ├── HerbaceousVegetation
│  ├── Highway
│  ├── ......
│
├──test/
│  ├── AnnualCrop
│  ├── Forest
│  ├── HerbaceousVegetation
│  ├── Highway
│  ├── ......
```

## Requirements
<img width="759" height="1218" alt="image" src="https://github.com/user-attachments/assets/836504b4-35cf-4eeb-8f2a-3a51b9545b05" />
<img width="703" height="1220" alt="image" src="https://github.com/user-attachments/assets/28f87c34-7265-4d83-adb9-291f095a4b85" />
<img width="609" height="468" alt="image" src="https://github.com/user-attachments/assets/c9b94e1b-dd9a-4120-b4e2-5e30b567b0f9" />

timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12; pyyaml; 

## Train
### Training on MSTAR-10classes
Setting hyper-parameters in MSTAR.yml

```
cd SAR
cd MSTAR-10classes-SAR
# 로그 파일 만들기
mkdir -p ./STDOUT ./STDERR
# 로그 파일 이름 지정
LOG_PREFIX="SAR_class.$(hostname).$(date +%Y%m%d_%H%M%S)"
# 명령어 실행
python train.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err 
```

### Testing MSTAR-10classes Val data
```
cd SAR
cd MSTAR-10classes-SAR
# 명령어 실행
python test.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err 
```

### Training on EuroSAT
Setting hyper-parameters in EuroSAT.yml
```
cd SAR
cd EuroSAT-SAR
# 로그 파일 만들기
mkdir -p ./STDOUT ./STDERR
# 로그 파일 이름 지정
LOG_PREFIX="SAR_class.$(hostname).$(date +%Y%m%d_%H%M%S)"
# 명령어 실행
python train.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err 
```

### Testing EuroSAT Val data
```
cd SAR
cd EuroSAT-SAR
# 명령어 실행
python test.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err 
```

### Energy Consumption Calculation
```
여기부터
cd imagenet
python energy_consumption_calculation_on_imagenet.py
```


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
