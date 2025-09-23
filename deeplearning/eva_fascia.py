import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm import tqdm
import os
import glob
from functools import partial
import multiprocessing
import numpy as np  
from torch.utils.data import TensorDataset, DataLoader
from test_dataload import test_dataload
import shap
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用配置文件中的路径
data = pd.read_csv(os.path.join(TEST_DIR, 'fascia_all_test.csv'), index_col=0)
# 转置数据，使得每一行代表一个细胞
data = data.transpose()
# 将索引作为labels
labels = data.index
labels
# 将索引作为labels
#这一步很重要，R的矩阵输出不能容忍相同标签的内容，所以会自动加序号！！！！！
labels = [label.split('.')[0] for label in data.index]
labels
# 创建一个LabelEncoder对象
encoder = LabelEncoder()
# 将细胞的barcodes转换为整数
encoded_labels = encoder.fit_transform(labels)
# 将所有的数字矩阵部分作为features
features = data.values
# 转换为torch.Tensor，并移动到GPU上
labels = torch.tensor(encoded_labels, device=device).long()  # 确保labels是Long类型
features = torch.tensor(features.astype(float), device=device)  # 确保features是浮点数
# 将数据分为训练集和测试集
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.9999, random_state=42) #全部用于测试
# 创建训练集和测试集的DataLoader
train_dataset = TensorDataset(features_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers=5, activation=nn.Sigmoid()):
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

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score

from os import listdir
from os.path import isfile, join

# 评估各种模型
models_to_test = [
    ('CPTC', 'fascia_CPTC', 128, 5, nn.ELU(), 'best_model_lr0.01_hs128_nl5_Adam_ELU_acc99.88.pth'),
    ('FC', 'fascia_FC', 128, 4, nn.ReLU(), 'best_model_lr0.01_hs128_nl4_Adam_ReLU_acc100.00.pth'),
    ('EC', 'fascia_EC', 31, 7, nn.Hardswish(), 'best_model_lr0.01_hs31_nl7_RMSprop_Hardswish_acc100.00.pth'),
    ('ETC', 'fascia_ETC', 44, 5, nn.ELU(), 'best_model_lr0.01_hs44_nl5_RMSprop_ELU_acc100.00.pth'),
    ('LP', 'fascia_LP', 128, 4, nn.ELU(), 'best_model_lr0.01_hs128_nl4_Adam_ELU_acc100.00.pth'),
    ('MC', 'fascia_MC', 128, 4, nn.ELU(), 'best_model_lr0.01_hs128_nl4_Adam_ELU_acc100.00.pth'),
    ('MuC', 'fascia_MuC', 128, 4, nn.Hardswish(), 'best_model_lr0.01_hs128_nl4_Adam_Hardswish_acc100.00.pth'),
    ('MuSC', 'fascia_MuSC', 128, 4, nn.ELU(), 'best_model_lr0.01_hs128_nl4_Adam_ELU_acc100.00.pth'),
    ('Others', 'fascia_Others', 128, 4, nn.ELU(), 'best_model_lr0.01_hs128_nl4_Adam_ELU_acc100.00.pth')
]

for name, folder, hidden_size, num_layers, activation, model_file in models_to_test:
    print(f"\n=== 评估 {name} 模型 ===")
    
    # 对于MuSC和Others，只有一个分类
    num_classes = 1 if name in ['MuSC', 'Others'] else len(torch.unique(labels))
    
    # 创建模型
    model = Net(input_size=features.shape[1], 
                num_hidden_layers=num_layers, 
                activation=activation, 
                hidden_size=hidden_size, 
                num_classes=num_classes)
    
    # 加载模型权重
    model_path = os.path.join(TRAIN_DIR, folder, model_file)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        # 评价
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                features_batch = features_batch.to(device)
                outputs = model(features_batch.float())
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.tolist())
                true_labels.extend(labels_batch.tolist())
        
        predicted_labels = torch.tensor(predicted_labels)
        true_labels = torch.tensor(true_labels)
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"模型文件不存在: {model_path}")

print("\n=== 评估完成 ===")
