# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl
import torchvision
import torchvision.models as models




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Step 1: Define the LungCancerModel class
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),    # This will work as RandomCrop after resizing
    transforms.RandomHorizontalFlip(),        # This applies to training images
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),  # CIFAR-100 mean and std
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

# Set device (GPU or CPU)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#Load the Teacher Model trained in teacher_client Code
net = models.densenet201(pretrained=False)
net.classifier = nn.Linear(net.classifier.in_features, 100)



# Step 2: Load the saved weights from the .npz file
npz_file_path = "./round-3-weights.npz"  # Replace with your file path
loaded = np.load(npz_file_path)

net = net.to(device)
model = net

# Step 4: Load the weights into the model's state_dict
state_dict = OrderedDict()
for i, (key, value) in enumerate(zip(model.state_dict().keys(), loaded.values())):
    state_dict[key] = torch.tensor(value)  # Convert np.ndarray to torch.Tensor

# Load the state_dict into the model
model.load_state_dict(state_dict)

print("Model loaded with weights from npz file.")

# Check if the weights are loaded correctly
for name, param in model.named_parameters():
    print(f"Layer: {name} | Weights: {param.data.cpu().numpy().flatten()[:5]}")  # Move tensor to CPU before converting to NumPy


# Check if the weights are loaded correctly
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Weights: {param.data.numpy().flatten()[:5]}")  # Printing first 5 weights of each layer for verification



teacher=model


# Convert to PyTorch tensors


student_model = models.mobilenet_v2(pretrained=False)

# Modify the classifier to have 100 output classes for CIFAR-100
student_model.classifier[1] = nn.Linear(student_model.classifier[1].in_features, 100)  # CIFAR-100 has 100 classes
  # CIFAR-10 has 10 classes
student_model.to(device)


for param in teacher.parameters():
    param.requires_grad = False


def distillation_loss(student_outputs, teacher_outputs, labels, T, alpha):
    """
    Compute the distillation loss using KL Divergence and Cross Entropy.
    :param student_outputs: Outputs from the student model
    :param teacher_outputs: Outputs from the teacher model
    :param labels: True labels
    :param T: Temperature for softening probabilities
    :param alpha: Weight for combining losses
    :return: Combined loss
    """
    # Soft targets loss
    criterion = nn.CrossEntropyLoss().to(device)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_outputs / T, dim=1),
                               nn.functional.softmax(teacher_outputs / T, dim=1)) * (T * T)
    # True labels loss
    hard_loss = criterion(student_outputs, labels)
    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss

def train_student(teacher_model, student_model, train_loader, epochs=10, T=2.0, alpha=0.5):
    teacher_model.to(device)
    teacher_model.eval()
   # Teacher model is in eval mode
    student_model.to(device)
    student_model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.01)# Student model is in training mode

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Get outputs from teacher model (no gradient computation)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            # Get outputs from student model
            student_outputs = student_model(images)

            # Compute the distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print epoch results
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# Example usage
# train_losses = train_student_with_distillation(teacher, net , train_loader, num_epochs=10)

def test(net, test_loader):
    """Validate the model on the test set."""
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
        return [val.cpu().numpy() for _, val in student_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(student_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        student_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_student(teacher, student_model , train_loader, epochs=15,T=2.0, alpha=0.5)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(student_model, test_loader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


fl.client.start_numpy_client(server_address="127.0.0.1:8083", client=CifarClient(),)

