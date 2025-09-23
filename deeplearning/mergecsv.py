#mergecev.py
import os
import pandas as pd
from multiprocessing import Pool, cpu_count

def merge_csv_files(folder_path):
    """
    合并指定文件夹中的所有CSV文件为一个文件。

    参数:
    folder_path (str): 包含CSV文件的文件夹路径。

    返回:
    None
    """
    # 获取文件夹中所有CSV文件的文件名列表
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 如果文件夹中有CSV文件
    if csv_files:
        # 创建一个空DataFrame来存储所有CSV文件的数据
        combined_data = pd.DataFrame()
        
        # 遍历每个CSV文件并合并数据
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            # 读取CSV文件的数据并添加到combined_data中
            df = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        
        # 将合并后的数据保存为一个新的CSV文件
        combined_data.to_csv(os.path.join(folder_path, "combined_data.csv"), index=False)
        print("合并完成，合并后的文件为combined_data.csv")
    else:
        print("文件夹中不存在CSV文件。")