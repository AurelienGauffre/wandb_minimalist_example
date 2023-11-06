# CIFAR-10 Training Showcase with Weights & Biases

This project demonstrates how to use Weights & Biases (wandb) for managing and tracking machine learning experiments. You will find a `train.py` script in this repository, which is designed to train a modified ResNet-18 model on the CIFAR-10 dataset with full integration of wandb for logging and sweeps.

## Prerequisites

To use this project, you will need the following:
- Python 3.6+
- PyTorch
- torchvision
- wandb
- OmegaConf

Install the necessary packages using pip:

```bash
pip install torch torchvision wandb omegaconf
```

Create an account on Weights & Biases and set up the wandb CLI on your machine.


## Wandb Classical run 

To start a standard training run with your configuration:

```bash
python train.py --config config.yaml
```


## Wandb Sweep run 

For a hyperparameter sweep, prepare a config_sweep.yaml file. Here's a brief example:

```bash
python train.py --config config_sweep.yaml
```
** The string'sweep'has to be inside the config sweep file name for the training script to know that it should use a sweep run.**


## Additional Resources
For a detailed explanation of wandb sweeps, refer to the official wandb documentation : https://docs.wandb.ai/guides/sweeps

