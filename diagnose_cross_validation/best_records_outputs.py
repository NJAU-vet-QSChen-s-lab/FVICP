import pandas as pd
import os
import glob
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
from config import *

# 创建一个空DataFrame来存储最佳模型参数
best_models_parameters = pd.DataFrame(columns=['Folder_Name', 'Learning_Rate', 'Hidden_Size', 'Optimizer', 'Activation', 'Accuracy'])

# 使用配置文件中的路径
folders = [
    os.path.join(CV_B1_DIR, 'fascia_b1_all_train'),
    os.path.join(CV_B2_DIR, 'fascia_b2_all_train')
]

for folder_path in folders:
    combined_data_path = os.path.join(folder_path, "combined_data.csv")
    
    if os.path.exists(combined_data_path):
        # 读取combined_data.csv文件
        combined_data = pd.read_csv(combined_data_path)
        
        # 找到Accuracy列数值最高的那一行
        max_accuracy_index = combined_data['Accuracy'].idxmax()
        max_accuracy_row = combined_data.loc[max_accuracy_index]
        
        # 获取文件夹名称
        folder_name = os.path.basename(folder_path)
        
        # 输出信息
        print(f"文件夹名称: {folder_name}")
        print("最高准确率对应的参数：")
        print(max_accuracy_row)
        print()  # 空行，用于分隔不同文件夹的输出
        
        # 添加最佳模型参数到DataFrame中
        best_models_parameters = pd.concat([best_models_parameters, pd.DataFrame({'Folder_Name': folder_name, 
                                                                                  'Learning_Rate': max_accuracy_row['Learning Rate'], 
                                                                                  'Hidden_Size': max_accuracy_row['Hidden Size'], 
                                                                                  "Layers": max_accuracy_row['Layers'],
                                                                                  'Optimizer': max_accuracy_row['Optimizer'], 
                                                                                  'Activation': max_accuracy_row['Activation'], 
                                                                                  'Accuracy': max_accuracy_row['Accuracy']}, 
                                                                                 index=[0])], 
                                           ignore_index=True)
    else:
        print(f"文件夹 {folder_path} 中的 combined_data.csv 文件不存在。")

# 输出所有最佳模型参数
print("\n所有最佳模型参数：")
print(best_models_parameters)

# 将最佳模型参数输出到CSV文件
best_models_parameters.to_csv('best_models_parameters.csv', index=False)

print("=== 交叉验证结果汇总完成 ===")
