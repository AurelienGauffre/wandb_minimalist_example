import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import wandb
from torchvision.models import resnet18

# Define the network architecture
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train(args=None, use_wandb_sweep=True):
    if use_wandb_sweep:
        run = wandb.init()
        args = run.config
    else:
        args = OmegaConf.create(args)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    model = ModifiedResNet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
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
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        metrics = {'epoch': epoch, 'train_loss': running_loss / len(trainloader), 'test_loss': test_loss / len(testloader), 'accuracy': accuracy}

        if use_wandb_sweep:
            wandb.log(metrics)

    if use_wandb_sweep:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args_command_line = parser.parse_args()

    USE_WANDB_SWEEP = "sweep" in args_command_line.config  # wandb sweep config files must contain "sweep" in their name
    config_dict = OmegaConf.to_container(OmegaConf.load(args_command_line.config), resolve=True)  # Convert the yaml file into a classical python dictionary

    if not USE_WANDB_SWEEP:
        train(config_dict, False)
    else:
        sweep_id = wandb.sweep(sweep=config_dict, project="cifar10_resnet18")
        print(f"Created a wandb Sweep with id: {sweep_id}")
        wandb.agent(sweep_id=sweep_id, function=train)

