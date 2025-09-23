import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

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

activation_functions = {
    "ReLU": nn.ReLU(),
    "Hardswish": nn.Hardswish(),
    "ELU": nn.ELU(),
    "Sigmoid": nn.Sigmoid(),
    "Softmax": nn.Softmax(dim=1)
}

# 使用配置文件中的路径
best_models_parameters = pd.read_csv(os.path.join(VISUALIZATION_DIR, 'best_models_parameters.csv'))
traindata_folder = os.path.join(CV_B1_DIR)
training_records = []

# 存储所有损失和准确率数据
all_train_losses = []
all_val_losses = []  # 存储验证损失
all_val_accuracies = []

for index, row in tqdm(best_models_parameters.iterrows(), desc="Training Models"):
    folder_name = row['Folder_Name']
    print(f"Starting training for {folder_name}")
    learning_rate = row['Learning_Rate']
    hidden_size = int(row['Hidden_Size'])
    num_hidden_layers = int(row['Layers'])
    #hidden_size = row['Hidden_Size']
    optimizer_name = row['Optimizer']
    activation_name = row['Activation']
    #num_hidden_layers = row['Layers']
    if activation_name in activation_functions:
        activation = activation_functions[activation_name]
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")
    csv_file = os.path.join(traindata_folder, f'{folder_name}.csv')
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file, index_col=0)
        data = data.transpose()
        labels = [label.split('.')[0] for label in data.index]
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        labels = torch.tensor(encoded_labels, device=device).long()
        features = data.values
        features = torch.tensor(features.astype(float), device=device)
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)
        train_dataset = TensorDataset(features_train, labels_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(features_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = Net(input_size=features.shape[1], num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, num_classes=len(torch.unique(labels)), activation=activation)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        early_stopping_epochs = 5
        early_stopping_counter = 0
        best_accuracy = 0
        train_losses = []
        val_losses = []  # 存储每个epoch的验证损失
        val_accuracies = []
        for epoch in range(1, 101):
            model.train()
            epoch_loss = 0
            for features_batch, labels_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(features_batch.float())
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for features_batch, labels_batch in test_loader:
                    features_batch = features_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = model(features_batch.float())
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()
                val_loss /= len(test_loader)
                val_losses.append(val_loss)
            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            print(f"Folder: {folder_name}, Epoch: {epoch}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}")
            scheduler.step()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                early_stopping_counter = 0
                if best_accuracy > 99:
                    torch.save(model.state_dict(), f'b1_{os.path.basename(csv_file)}_{best_accuracy:.2f}.pth')
                    print(f"Completed Epoch {epoch} for {folder_name}")
            else:
                early_stopping_counter += 1
            if early_stopping_counter > 9:
                print("No improvement in accuracy for 10 epochs, stopping training.")
                print(f"Completed Epoch {epoch} for {folder_name}")
                break
        model_save_path = f"b1_{folder_name}_accuracy_{best_accuracy:.2f}_model.pth"
        torch.save(model.state_dict(), model_save_path)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)  # 存储验证损失
        all_val_accuracies.append(val_accuracies)
        training_records.append([folder_name, epoch, accuracy])

# Plot all training losses, validation losses, and validation accuracies on the same figure
plt.figure(figsize=(15, 5)) #宽度,长度

# Plot training losses
for i, train_losses in enumerate(all_train_losses):
    plt.subplot(1, 3, 1) # 修改为3个子图，一行三列
    plt.plot(train_losses, label=f'{best_models_parameters.iloc[i]["Folder_Name"]} Training Loss')

plt.subplot(1, 3, 1)
plt.title('Training Loss for All Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot validation losses
for i, val_losses in enumerate(all_val_losses):
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label=f'{best_models_parameters.iloc[i]["Folder_Name"]} Validation Loss')

plt.subplot(1, 3, 2)
plt.title('Validation Loss for All Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot validation accuracies
for i, val_accuracies in enumerate(all_val_accuracies):
    plt.subplot(1, 3, 3)
    plt.plot(val_accuracies, label=f'{best_models_parameters.iloc[i]["Folder_Name"]} Validation Accuracy')

plt.subplot(1, 3, 3)
plt.title('Validation Accuracy for All Models')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('b1_all_models_training_visualization_legend.pdf')
plt.close()

training_records_df = pd.DataFrame(training_records, columns=['Folder_Name', 'Epoch', 'Accuracy'])
training_records_df.to_csv('training_records.csv', index=False)

# Plot all training losses, validation losses, and validation accuracies on the same figure
plt.figure(figsize=(15, 5)) #宽度,长度

# Plot training losses
for i, train_losses in enumerate(all_train_losses):
    plt.subplot(1, 3, 1) # 修改为3个子图，一行三列
    plt.plot(train_losses, label=f'{best_models_parameters.iloc[i]["Folder_Name"]} Training Loss')

plt.subplot(1, 3, 1)
plt.title('Training Loss for All Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.legend()

# Plot validation losses
for i, val_losses in enumerate(all_val_losses):
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label=f'{best_models_parameters.iloc[i]["Folder_Name"]} Validation Loss')

plt.subplot(1, 3, 2)
plt.title('Validation Loss for All Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.legend()

# Plot validation accuracies
for i, val_accuracies in enumerate(all_val_accuracies):
    plt.subplot(1, 3, 3)
    plt.plot(val_accuracies, label=f'{best_models_parameters.iloc[i]["Folder_Name"]} Validation Accuracy')

plt.subplot(1, 3, 3)
plt.title('Validation Accuracy for All Models')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.legend()

plt.tight_layout()
plt.savefig('b1_all_models_training_visualization.pdf')
plt.close()

training_records_df = pd.DataFrame(training_records, columns=['Folder_Name', 'Epoch', 'Accuracy'])
training_records_df.to_csv('b1_training_records.csv', index=False)

print("=== 可视化完成 ===")