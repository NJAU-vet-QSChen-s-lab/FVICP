import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

from mergecsv import merge_csv_files

def parallel_merge_csv(folder_paths, num_processes=None):
    """
    使用多核心并行处理合并CSV文件的任务。

    参数:
    folder_paths (list): 包含多个CSV文件的文件夹路径列表。
    num_processes (int): 进程池中的进程数，默认为None，表示使用计算机可用的核心数。

    返回:
    None
    """
    # 如果没有指定进程数，则使用计算机可用的核心数
    if num_processes is None:
        num_processes = cpu_count()
    
    # 创建一个进程池
    with Pool(processes=num_processes) as pool:
        # 使用进程池并行处理合并CSV文件的任务
        pool.map(merge_csv_files, folder_paths)

# 使用配置文件中的路径构建文件夹路径列表
folder_names = [
    "fascia_CPTC", 
    "fascia_FC", 
    "fascia_EC", 
    "fascia_ETC", 
    "fascia_MC", 
    "fascia_MuC", 
    "fascia_MuSC", 
    "fascia_LP", 
    "fascia_Others"
]

folder_paths = [os.path.join(TRAIN_DIR, folder_name) for folder_name in folder_names]

# 调用并行合并CSV文件的函数
parallel_merge_csv(folder_paths)

print("=== 多文件合并完成 ===")
