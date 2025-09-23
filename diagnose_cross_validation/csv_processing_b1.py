import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import csv
import os
from itertools import product

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers=5, activation=nn.ReLU()):
        super(Net, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size, num_classes))
        self.activation = activation

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)
        x = self.fc[-1](x)
        return x

criterion = nn.CrossEntropyLoss()

learning_rates = np.arange(0.01, 0.1, 0.01)
hidden_sizes = [64, 128] #np.arange(20, 60, 1)
layer_numbers = np.arange(4, 10, 1)  # Adding range for number of layers

optimizers = [optim.SGD, optim.Adam, optim.RMSprop]
activations = [nn.Sigmoid(), nn.ReLU(), nn.ELU(), nn.Softmax(dim=1), nn.Hardswish()]

def train_model(params, train_loader, test_loader, features_shape, labels_unique, result_folder):
    lr, hidden_size, num_layers, opt, act = params
    model = Net(input_size=features_shape, hidden_size=hidden_size, num_classes=labels_unique, num_hidden_layers=num_layers, activation=act)
    model.to(device)
    optimizer = opt(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    num_epochs = 50
    best_accuracy = 0.0
    early_stop_counter = 0
    for epoch in range(num_epochs):
        model.train()
        for features_batch, labels_batch in train_loader:
            outputs = model(features_batch.float())
            loss = criterion(outputs, labels_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                outputs = model(features_batch.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch.long()).sum().item()
        accuracy = 100 * correct / total
        print(f"Learning Rate: {lr}, Hidden Size: {hidden_size}, Layers: {num_layers}, "
              f"Optimizer: {opt.__name__}, Activation: {type(act).__name__}, "
              f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")
        scheduler.step()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop_counter = 0
            if best_accuracy > 99:
                #torch.save(model.state_dict(), os.path.join(result_folder, f'best_model_{best_accuracy:.2f}.pth'))
                torch.save(model.state_dict(), os.path.join(result_folder, f'best_model_lr{lr}_hs{hidden_size}_nl{num_layers}_{opt.__name__}_{type(act).__name__}_acc{best_accuracy:.2f}.pth'))
        else:
            early_stop_counter += 1
        if early_stop_counter > 9 or accuracy >= 99.9:
            break
    return best_accuracy

def process_csv(csv_file):
    data = pd.read_csv(csv_file, index_col=0)
    data = data.transpose() #R语言输出的脚本做一下转置
    labels = [label.split('.')[0] for label in data.index] #R语言输出带着.0.1这样的数字序号，需要删掉
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    features = data.values
    labels = torch.tensor(encoded_labels, device=device).long()
    features = torch.tensor(features.astype(float), device=device)
    labels_unique = len(torch.unique(labels))
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    result_folder = os.path.splitext(os.path.basename(csv_file))[0]
    os.makedirs(result_folder, exist_ok=True)

    param_grid = product(learning_rates, hidden_sizes, layer_numbers, optimizers, activations)

    for params in param_grid:
        best_accuracy = train_model(params, train_loader, test_loader, features.shape[1], labels_unique, result_folder)
        lr, hidden_size, num_layers, opt, act = params
        result_file = os.path.join(result_folder, f'{lr}_{hidden_size}_{num_layers}_{opt.__name__}_{type(act).__name__}_result.csv')
        with open(result_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Learning Rate", "Hidden Size", "Layers", "Optimizer", "Activation", "Accuracy"])
            writer.writerow([lr, hidden_size, num_layers, opt.__name__, type(act).__name__, best_accuracy])