import os
import argparse
from datetime import datetime
from typing import Dict
from pprint import pprint
from ray.air import session
from ray.air import Checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

cur_path = os.path.dirname(os.path.realpath(__file__))
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="~/data",
    train=True,
    download=True,
    transform=transform,
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="~/data",
    train=False,
    download=True,
    transform=transform,
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) // session.get_world_size()
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // session.get_world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y, reduction='sum').item()
            pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test: "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // session.get_world_size()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=worker_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=worker_batch_size)

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # Create model.
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    loss_fn = F.nll_loss
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        loss = validate_epoch(test_dataloader, model, loss_fn)
        session.report(dict(loss=loss))
        session.report({"loss": loss,
                        "epoch": epoch
                        },
                        checkpoint=Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict()))
                    )
        # state_dict = model.state_dict()
        # consume_prefix_in_state_dict_if_present(state_dict, "module.")
        # train.save_checkpoint(epoch=epoch, model_weights=state_dict)


def train_mnist(num_workers=2, use_gpu=False):
    trainer = TorchTrainer(
        train_loop_per_worker=train_func, # train function
        train_loop_config={"lr": 1.0, "batch_size": 64, "epochs": 12}, # parameter
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")
    print(f"best_checkpoints: {result.best_checkpoints}")
    print(f"result.log_dir: {result.log_dir}")
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    result.checkpoint.to_directory("models/"+dt_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args, _ = parser.parse_known_args()

    import ray

    if args.smoke_test:
        # 2 workers + 1 for trainer.
        ray.init(num_cpus=3)
        train_mnist()
    else:
        ray.init(address=args.address)
        train_mnist(num_workers=args.num_workers, use_gpu=args.use_gpu)

