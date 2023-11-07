import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import random
from omegaconf import OmegaConf
import wandb

DATASET_FRACTION = .2  # take 20% of the dataset to speed up training


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Cifar, self).__init__()
        self.resnet = resnet18(pretrained=False)
        # Classical way to adjust the first convolutional layer for small images
        # since the original ResNet18 is trained on ImageNet
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def train(config=None):
    if config is None:  # it means we are using a wandb sweep, so the config has to be loaded from wandb
        run = wandb.init()
        config = run.config
    else:  # it means we are using directly the given config file, and we upload it to wandb to keep a record of the training parameters
        config = OmegaConf.create(config)
        wandb.init(project="cifar10_resnet18",
                   config=config)  # Add as argument name=config.run_name if you want to name the run

    print(f"Training on {DATASET_FRACTION * 100}% of Cifar10 dataset")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load the full CIFAR-10 dataset and select a subset of it
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset_indices_train = random.sample(range(len(full_trainset)), int(DATASET_FRACTION * len(full_trainset)))
    subset_indices_test = random.sample(range(len(full_testset)), int(DATASET_FRACTION * len(full_testset)))
    trainset = Subset(full_trainset, subset_indices_train)
    testset = Subset(full_testset, subset_indices_test)
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = ResNet18Cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        metrics = {'epoch': epoch, 'train/loss': running_loss / len(trainloader),
                   'test/loss': test_loss / len(testloader), 'test/accuracy': accuracy}
        print(metrics)
        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args_command_line = parser.parse_args()

    USE_WANDB_SWEEP = "sweep" in args_command_line.config
    config_dict = OmegaConf.to_container(OmegaConf.load(args_command_line.config), resolve=True)

    if not USE_WANDB_SWEEP:
        train(config_dict)
    else:
        sweep_id = wandb.sweep(sweep=config_dict, project="cifar10_resnet18")
        print(f"Created a wandb Sweep with id: {sweep_id}")
        wandb.agent(sweep_id=sweep_id, function=train)
