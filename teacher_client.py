# -*- coding: utf-8 -*-




from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import torchvision.models as models
from torch.utils.data import DataLoader, random_split

# Define transformations for data normalization
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize to 224x224
    transforms.RandomCrop(224, padding=4),       # Random crop with padding
    transforms.RandomHorizontalFlip(),           # Random horizontal flip
    transforms.RandomRotation(15),               # Random rotation in the range [-15, 15] degrees
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random changes in brightness, contrast, saturation, and hue
    transforms.RandomGrayscale(p=0.1),           # Convert to grayscale with a probability of 10%
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation (translation)
      # Randomly erasing parts of an image
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),     # Normalizing using CIFAR-100 mean and std
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),  # CIFAR-100 mean and std
])

# Load ImageNet dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


num_examples = {"trainset" : len(train_dataset), "testset" : len(test_dataset)}
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Create DataLoaders

# Set device (GPU or CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load Teacher model according to your needs
net = models.densenet201(pretrained=False)  # Set pretrained=True to use pre-trained weights
net.classifier = nn.Linear(net.classifier.in_features, 100)  # Adjust final layer for CIFAR-100 classes



# Move the model to the device (GPU or CPU)
net = net.to(device)
# Print model summary
print(net)
def train(net, train_loader, epochs):
    """Train the model on the training set."""
    net.to(device)  # Move model to GPU if available
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)  # Using SGD for better results
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Cosine Annealing LR scheduler

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Adjust the learning rate
        scheduler.step()


def test(net, test_loader):
    """Validate the model on the test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss().to(device)  # Ensure criterion is on the same device
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()  # Sum the loss for averaging later

            # Get predictions and calculate correct predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(test_loader)  # Average loss over all batches
    accuracy = 100. * correct / total  # Correct predictions percentage

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


class CifarClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=15)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_loader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


fl.client.start_numpy_client(server_address="127.0.0.1:8082", client=CifarClient(),)