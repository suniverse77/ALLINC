# All-Around Neural Collapse for Imbalanced Classification

**Unofficial code**

This code is implementing paper [All-Around Neural Collapse for Imbalanced Classification](https://arxiv.org/abs/2408.07253).

Base model is ResNet and used dataset is CIFAR10.

&nbsp;
## Train

### Run

```bash train.sh <mode> <run_name> <GPU device num>```

### Arguments

**mode**
- ```all``` (w/AllINC)
- ```basic``` (w/o AllINC)

**run_name:** The name of the training run. (e.g., training date)

**GPU device num:** The device number of the GPU to be used for model training.

&nbsp;
## Results

Performance in terms of Top-1 Accuracy:

| Imbalance Ratio   | 200       | 100       | 50        | 10        | 1         |
|:----------------- |:--------- |:--------- |:--------- |:--------- |:--------- |
| CE Loss           | 38.91     | 44.00  	| 53.51 	| 64.46     | 79.70     |
| AllINC            | **49.51** | **50.81** | **60.04** | **75.91** | **80.64** |
