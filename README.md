# PushPull
Implementation for the linear speedup property of PushPull algorithm on neural network.

## Setup

1. Download the MNIST or CIFAR10 dataset and unzip then in the directory `./data`.
2. Change the directory of the datasets in `./datasets/prepare_data.py`.
3. Change the directory of the output files in `./training/training_track_grad_norm.py`.
4. To set up the environment, follow these steps:
```bash
conda create -n pushpull python=3.12
source activate pushpull
pip install -r requirements.txt
```

## Demo

In the `./demo.py`, you can set the parameters like follows, we give a demo on the grid network topology.
```python
n = 4
lr = 3e-2
num_epochs = 100
bs = 128
alpha = 0.9
use_hetero = True
remark = "new"
device = "cuda:0"
root = "./output"
```
you can also change the network topology based on the functions in the `./utils/network_utils.py`.