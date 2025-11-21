# SAR-Spikingformer: Adaptive Spiking Transformer for SAR Image Classification, [AAAI 2026 Workshop on Neuromorphic Intelligence](https://openreview.net/pdf?id=98d4g2w0qc)
Based on Spikingformer, this study proposes a lightweight spiking transformer **SAR-Spikingformer** optimized for the classification of synthetic aperture radar (SAR) images.

SAR-Spikingformer introduces SAR-specific data augmentation and a dynamic 'block-halting' mechanism to significantly reduce energy consumption, the key strength of SNN, by up to **27.76%** and **46.53%**, respectively, while maintaining classification accuracy on MSTAR and EuroSAT datasets.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/64926ecf-a90f-46ad-9d5f-2288e0e216fa" />


## Datasets
data prepare: data with the following folder structure
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
│  ├── HerbaceousVegatation
│  ├── Highway
│  ├── ......
│
├──test/
│  ├── AnnualCrop
│  ├── Forest
│  ├── HerbaceousVegatation
│  ├── Highway
│  ├── ......
```

## Requirements
timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12; pyyaml; 

## Train
### Training on MSTAR-10classes
Setting hyper-parameters in MSTAR-10classes.yml
```
cd SAR
cd MSTAR-10classes-SAR

# Create a log file
mkdir -p ./STDOUT ./STDERR

# Name the log file
LOG_PREFIX="SAR_class.$(hostname).$(date +%Y%m%d_%H%M%S)"

# Run the command
python train.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
```

### Training on EuroSAT
Setting hyper-parameters in EuroSAT.yml
```
cd SAR
cd EuroSAR-SAR

# Create a log file
mkdir -p ./STDOUT ./STDERR

# Name the log file
LOG_PREFIX="SAR_class.$(hostname).$(date +%Y%m%d_%H%M%S)"

# Run the command
python train.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
```

## Test & Energy Consumption
### Testing MSTAR-10classes
```
cd SAR
cd MSTAR-10classes-SAR
python test.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
```

### Testing EuroSAT
```
cd SAR
cd EuroSAT-SAR
python test.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
```

### Energy Consumption Calculation
```
cd SAR
cd MSTAR-10classes-SAR
python energy_consumption_calculation.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
```
```
cd SAR
cd EuroSAT-SAR
python energy_consumption_calculation.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
```

## Result
### Main results on MSTAR-10classes
#### Architecture 8-384
| Model               | Param (M)| T |  MACs (G)     | ACs (G)   |  Energy Consumption (mJ) |Top-1 Acc|
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |
| spikformer | 16.44   | 4 |  2.20     | 0.42   | 10.50   |89.50%  |
| Spikingformer | 16.43   | 4 |  0.08     | 0.91  | 1.21   |93.19%  |
| Ours | 16.45   | 4 |  0.08     | 0.54 (-40.9%)  |  0.87 (-27.76%)  |93.28%  |

#### Architecture 8-512
| Model               | Param (M)| T |  MACs (G)     | ACs (G)   |  Energy Consumption (mJ) |Top-1 Acc|
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |
| spikformer | 29.19   | 4 |  3.87     | 0.73   | 18.48   |89.68%  |
| Spikingformer | 29.17   | 4 |  0.11     | 1.61  | 1.97   |93.23%  |
| Ours | 29.21   | 4 |  0.11     | 0.91 (-43.78%)  |  1.33 (-27.76%)  |93.28%  |


### Main results on EuroSAT
#### Architecture 8-384
| Model               | Param (M)| T |  MACs (G)     | ACs (G)   |  Energy Consumption (mJ) |Top-1 Acc|
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |
| spikformer | 16.44   | 4 |  0.55     | 0.08   | 2.60   |98.20%  |
| Spikingformer | 15.10   | 4 |  0.02     | 0.52  | 0.57   |99.17%  |
| Ours | 15.13   | 4 |  0.02     | 0.27 (-48.20%)  |  0.34 (-39.94%)  |99.31%  |
#### Architecture 8-512
| Model               | Param (M)| T |  MACs (G)     | ACs (G)   |  Energy Consumption (mJ) |Top-1 Acc|
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |
| spikformer | 29.19   | 4 |  0.96     | 0.13   | 4.57   |98.51%  |
| Spikingformer | 26.81   | 4 |  0.03     | 0.89  | 0.93   |99.35%  |
| Ours | 26.85   | 4 |  0.03     | 0.41 (-54.02%)  |  0.50 (-46.53%)  |99.39%  |


## Reference
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
