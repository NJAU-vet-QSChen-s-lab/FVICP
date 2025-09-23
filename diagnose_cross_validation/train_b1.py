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
csv_files = glob.glob(os.path.join(CV_B1_DIR, '*.csv'))

num_cores = multiprocessing.cpu_count()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    #pool = multiprocessing.Pool(processes=num_cores)
    pool = multiprocessing.Pool(processes=15)
    # Process the CSV files in parallel
    partial_process_csv = partial(process_csv)
    pool.map(partial_process_csv, csv_files)
    pool.close()
    pool.join()

print("=== 交叉验证训练完成 ===")
