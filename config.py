# config.py

import random

# --- 数据集和客户端配置 (Dataset and Client Configuration) ---
NUM_CLIENTS = 10  # 客户端数量 (可以减少以便快速运行) (Number of clients (can reduce for quicker run))
CLASSES_PER_CLIENT = 5  # 每个客户端的类别数量 (N-way) (N-way (number of classes per client))
SHOTS_PER_CLASS = 19  # 每个类别在每个客户端上的训练样本数 (K-shot) (K-shot (training samples per class on each client))
                            # Omniglot每个类别有20个样本，因此 K 最大为 19 (支持集)，剩余的用作查询集
                            # (Omniglot has 20 samples per class, so K can be at most 19 for support, leaving rest for query)

# --- 随机种子 (Random Seed) ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED) # 为Python内置的random模块设置种子 (Set seed for Python's built-in random module)
# 注意: PyTorch的随机种子将在main.py中设置，以确保跨文件的一致性
# Note: PyTorch's random seed will be set in main.py for consistency across files

# --- 图像转换配置 (Image Transformation Configuration) ---
IMAGE_SIZE = (28, 28) # 图像大小调整目标 (Target image resize dimensions)

# --- 联邦学习训练配置 (Federated Learning Training Configuration) ---
ROUNDS = 20  # 联邦学习的轮数 (Number of federated learning rounds)
LEARNING_RATE = 0.01  # 客户端本地训练的学习率 (Learning rate for client local training)

# --- FsFL 特定配置 (FsFL Specific Configuration) ---
FSFL_TEMPERATURE = 2.0  # FsFL 中知识蒸馏的温度参数 (Temperature for knowledge distillation in FsFL)
FSFL_ALPHA = 0.5  # FsFL 中原型损失和蒸馏损失的权重因子 (Weight factor for prototype loss vs distillation loss in FsFL)

# --- 模型输出类别数 (Model Output Classes) ---
# 将在 main.py 中根据 NUM_CLIENTS 和 CLASSES_PER_CLIENT 动态计算
# Will be dynamically calculated in main.py based on NUM_CLIENTS and CLASSES_PER_CLIENT
# NUM_CLASSES_TOTAL = NUM_CLIENTS * CLASSES_PER_CLIENT

# --- 数据集根目录 (Dataset Root Directory) ---
DATA_ROOT = 'data' # Omniglot 数据集下载和存储的根目录 (Root directory for Omniglot dataset download and storage)