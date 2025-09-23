# =============================================================================
# 配置文件：路径设置
# =============================================================================
# 注意：请根据您的实际环境修改以下路径

import os

# 项目根目录
PROJECT_ROOT = "."

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# PyTorch相关目录
PYTORCH_DIR = os.path.join(OUTPUT_DIR, "pytorch")
TRAIN_DIR = os.path.join(PYTORCH_DIR, "train")
TEST_DIR = os.path.join(PYTORCH_DIR, "test")
MODEL_DIR = os.path.join(PYTORCH_DIR, "models")

# 交叉验证目录
CV_DIR = os.path.join(OUTPUT_DIR, "cross_validation")
CV_B1_DIR = os.path.join(CV_DIR, "b1")
CV_B2_DIR = os.path.join(CV_DIR, "b2")

# 可视化目录
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualization")

# 创建必要的目录
for dir_path in [DATA_DIR, OUTPUT_DIR, PYTORCH_DIR, TRAIN_DIR, TEST_DIR, 
                 MODEL_DIR, CV_DIR, CV_B1_DIR, CV_B2_DIR, VISUALIZATION_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 深度学习超参数设置
DL_PARAMS = {
    "learning_rates": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    "hidden_sizes": [64, 128],
    "num_hidden_layers": [4, 5, 6, 7],
    "optimizers": ["SGD", "Adam", "RMSprop"],
    "activations": ["ReLU", "Sigmoid", "ELU", "Softmax", "Hardswish"],
    "batch_size": 32,
    "early_stopping_patience": 10,
    "lr_scheduler_step": 30,
    "lr_scheduler_gamma": 0.1
}

# 打印配置信息
print("=== 项目配置信息 ===")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据目录: {DATA_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"PyTorch目录: {PYTORCH_DIR}")
print(f"训练目录: {TRAIN_DIR}")
print(f"测试目录: {TEST_DIR}")
print(f"模型目录: {MODEL_DIR}")
print(f"交叉验证目录: {CV_DIR}")
print(f"可视化目录: {VISUALIZATION_DIR}")
print("==================") 