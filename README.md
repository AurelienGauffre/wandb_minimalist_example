# CIFAR-10 Training Showcase with Weights & Biases

This repo demonstrates a simple example on how to use Weights & Biases (wandb) for managing and tracking machine learning experiments. You will find a `train.py` script in this repository, which is designed to train a ResNet-18 model on the CIFAR-10 dataset with full integration of wandb for logging and sweeps. 

## Prerequisites

To use this project, you will need the following:
- Python 3.6+
- PyTorch
- torchvision
- wandb
- OmegaConf (use to simply convert a yaml config file into a python object where attributes being the parameters, instead of using a python dict, ie. we can use config.lr instead of config['lr'])

Install the necessary packages using pip:

```bash
pip install torch torchvision wandb omegaconf
```

Create an account on Weights & Biases and set up the wandb CLI on your machine.


## Wandb classical run 

To start a standard training run with your configuration:

```bash
python train.py --config config.yaml
```


## Wandb Sweep run 

For a hyperparameter sweep, use the config_sweep.yaml file instead :

```bash
python train.py --config config_sweep.yaml
```
**As it's implemented in train.py, the string 'sweep' has to be inside the config sweep file name for the training script to know that it should use a sweep run.** This allows some flexibility by using the same script train.py for both classical trainning with wandb or wandb sweeps.


## Additional Resources
For a detailed explanation see the official doc of wandb :
* quick start with wandb : https://docs.wandb.ai/quickstart
* sweeps : https://docs.wandb.ai/guides/sweeps

