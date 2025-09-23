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
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型架构
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

# 读取最佳参数
best_models_parameters = pd.read_csv(os.path.join(TRAIN_DIR, 'best_models_parameters.csv'))

# 存储训练数据的文件夹路径
traindata_folder = os.path.join(TRAIN_DIR, "train_data")

# 存储训练记录的列表
training_records = []

# Define a mapping for activation functions
activation_functions = {
    "ReLU": nn.ReLU(),
    "Hardswish": nn.Hardswish(),
    "ELU": nn.ELU(),
    "Sigmoid": nn.Sigmoid(),
    "Softmax": nn.Softmax(dim=1)  # Softmax is usually applied on outputs, specify dim=1 for prediction
    # Add other activations if needed
}

# 循环训练模型
for index, row in tqdm(best_models_parameters.iterrows(), desc="Training Models"):
    folder_name = row['Folder_Name']
    learning_rate = row['Learning_Rate']
    hidden_size = row['Hidden_Size']
    optimizer_name = row['Optimizer']
    activation_name = row['Activation']
    # Fetch activation function
    if activation_name in activation_functions:
        activation = activation_functions[activation_name]
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")
    # 构建完整的文件路径
    csv_file = os.path.join(traindata_folder, f'{folder_name}.csv')
    # 加载数据
    if os.path.exists(csv_file):
        # 加载数据
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
        # Construct the model
        num_hidden_layers = 5  # Specify this if it's constant, or pass dynamically if you use different values
        model = Net(input_size=features.shape[1], hidden_size=hidden_size, num_classes=len(torch.unique(labels)), num_hidden_layers=num_hidden_layers, activation=activation)
        model.to(device)
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # 初始化早停参数
        early_stopping_epochs = 5
        early_stopping_counter = 0
        best_accuracy = 0
        # 训练模型
        for epoch in range(1, 51):  # 这里从1开始计数更直观
            model.train()
            for i, (features_batch, labels_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(features_batch.float())
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            # 在测试集上评估模型
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features_batch, labels_batch in test_loader:
                    features_batch = features_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = model(features_batch.float())
                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()
            accuracy = 100 * correct / total
            # 记录训练记录
            training_records.append([folder_name, epoch, accuracy])
            print(f"Folder: {folder_name}, Epoch: {epoch}, Accuracy: {accuracy:.2f}")            
            scheduler.step()
            # 判断是否需要早停
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                early_stop_counter = 0
                if best_accuracy > 99:
                    torch.save(model.state_dict(), f'{os.path.basename(csv_file)}_{best_accuracy:.2f}.pth')
            else:
                early_stop_counter += 1
            if early_stop_counter > 9:
                print("No improvement in accuracy for 10 epochs, stopping training.")
                break
        # 保存模型，并在文件名中包含最后一个epoch的准确率
        model_save_path = f"{folder_name}_accuracy_{best_accuracy:.2f}_model.pth"
        torch.save(model.state_dict(), model_save_path)
    else:
        print(f"文件 {csv_file} 不存在！")

# 将训练记录转换为DataFrame并保存为CSV文件
training_records_df = pd.DataFrame(training_records, columns=['Folder_Name', 'Epoch', 'Accuracy'])
training_records_df.to_csv('training_records.csv', index=False)

print("=== GPU训练完成 ===")
                            
