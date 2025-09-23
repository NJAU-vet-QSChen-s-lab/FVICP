import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
#import torch.optim as optim
#import csv
#from tqdm import tqdm
#import os
#import glob
#from functools import partial
#import multiprocessing
import numpy as np  
from torch.utils.data import TensorDataset, DataLoader
#from test_dataload import test_dataload
import shap
from torch.utils.data import DataLoader
#from sklearn.preprocessing import StandardScaler
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 重新加载包含基因名称的原始测试数据
# 包括基因名称（作为索引）和分组标签
original_test_data = pd.read_csv(os.path.join(TEST_DIR, 'fascia_all_test.csv'), index_col=0)

# 提取基因名称
feature_names = original_test_data.index.tolist()
feature_names

# 加载CSV文件
data = original_test_data.reset_index(drop=True)

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
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers=6, activation=nn.ReLU()):
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

#test_dataset = TensorDataset(features_test, labels_test)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from os import listdir
from os.path import isfile, join

##设置模型目录
path = os.path.join(TRAIN_DIR, 'fascia_all_test')
#获取目录内.pth后缀文件
files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.pth')]
files

labels_unique = len(torch.unique(labels))

model = Net(input_size=features.shape[1], hidden_size=64, activation=nn.ReLU(), num_hidden_layers=4, num_classes= labels_unique) #len(torch.unique(labels))) #nn.Sigmoid(), nn.ReLU(), nn.ELU(), nn.Softmax(dim=1), nn.Hardswish()
model.load_state_dict(torch.load(os.path.join(path, 'best_model_lr0.01_hs64_nl4_SGD_ELU_acc99.97.pth')))
model.to(device)
model.eval()

#初步评价
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
print("Accuracy: {:.2f}%".format(accuracy * 100))
#Accuracy: 99.97%

# 使用SHAP来解释模型

# 确保features_test和labels_test的类型都是浮点数
features_test = features_test.to(torch.float32, non_blocking=True)
labels_test = labels_test.to(torch.long, non_blocking=True)

# 创建一个TensorDataset
test_dataset = TensorDataset(features_test, labels_test)

# 创建一个DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建一个SHAP解释器
explainer = shap.DeepExplainer(model, test_dataset.tensors[0])

# 计算SHAP值
#shap_values = explainer.shap_values(test_dataset.tensors[0])
shap_values = explainer.shap_values(test_dataset.tensors[0], check_additivity=False)

# Average the SHAP values over the class dimension
shap_values_mean = np.mean(np.abs(shap_values), axis=-1)  # Average over the last dimension

# Calculate the mean absolute SHAP values across samples
feature_shap_values = shap_values_mean.mean(axis=0)  # Average over the samples

# Ensure the final shape is correct
if feature_shap_values.ndim == 1 and len(feature_names) == feature_shap_values.shape[0]:
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': feature_shap_values
    })
else:
    raise ValueError("Mismatch in dimensions: feature_names and feature_shap_values must be 1-dimensional and of the same length.")

print(feature_importance_df)

# 按照SHAP值对特征进行排序
feature_importance_df.sort_values(by='shap_value', ascending=False, inplace=True)

# 输出排序后的特征重要性
print(feature_importance_df)

# 如果需要，可以将结果保存到CSV文件
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'all_feature_importance.csv'))

# 计算召回率和F1分数
# 假设模型已经加载并评估，现在要评估模型的预测结果
# 获取模型的预测结果
predictions = model(features_test)
