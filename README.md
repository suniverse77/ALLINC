# All-Around Neural Collapse for Imbalanced Classification

**Unofficial code**

This code is implementing paper [All-Around Neural Collapse for Imbalanced Classification](https://arxiv.org/abs/2408.07253).

Base model is ResNet.

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
