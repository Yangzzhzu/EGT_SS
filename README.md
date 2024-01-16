# EGT-SS: Edge-Guided Transformer for Semantic Segmentation
### Introduction

We kindly provide the code for utilizing the ResNet101 model as mentioned in our publicly available research paper

<img src="./Framework.png" width="900"/>

### Environment

- Anaconda3
- Python == 3.7.9
- PyTorch == 1.7.1
- CUDA ==11.0

### Getting Started

```
git clone https://github.com/Yangzzhzu/EGT_SS.git
cd EGT_SS
conda create -n EGT_SS python=3.7
conda activate EGT_SS
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

### Prepare Datasets
- For Cityscapes, you can download from [Cityscapes](https://www.cityscapes-dataset.com/)

- For ADE20K, you can download from [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

### Train

Cityscapes
```
sh tool/train.sh cityscapes [EGT_SS101]
```

ADE20K
```
sh tool/trainade.sh ade20k [EGT_SS101]
```

### Evaluation

Validation on Cityscapes
```
sh tool/test.sh cityscapes [EGT_SS101]
```

Test on Cityscapes
```
sh tool/test.sh cityscapes [EGT_SS101]
```

Validation on ADE20K
```
sh tool/testade.sh ade20k [EGT_SS101]
```
