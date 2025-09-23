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
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

from csv_processing_1_fine import process_csv 

# 使用配置文件中的路径
csv_files = glob.glob(os.path.join(TRAIN_DIR, '*.csv'))

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

print("=== 精细训练完成 ===")
