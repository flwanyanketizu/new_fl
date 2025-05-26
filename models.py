# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    用于特征提取的卷积神经网络编码器。
    Convolutional Neural Network encoder for feature extraction.
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # 定义卷积层和批归一化层
        # Define convolutional and batch normalization layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 输入: [批大小, 1, 28, 28] (Input: [batch, 1, 28, 28])
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(self, x):
        # x 输入: [批大小, 1, 28, 28] (Input x: [batch, 1, 28, 28])
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # -> [批大小, 32, 14, 14] (-> [batch, 32, 14, 14])
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # -> [批大小, 32, 7, 7] (-> [batch, 32, 7, 7])
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # -> [批大小, 32, 3, 3] (-> [batch, 32, 3, 3])
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # -> [批大小, 32, 1, 1] (-> [batch, 32, 1, 1])
        
        # 展平为特征向量 [批大小, 32]
        # Flatten to feature vector [batch, 32]
        x = x.view(x.size(0), -1)
        return x

class FSFLModel(nn.Module):
    """
    包含分类器的完整模型 (用于 FedAvg)。
    Full model including classifier (for FedAvg).
    """
    def __init__(self, num_classes):
        super(FSFLModel, self).__init__()
        self.encoder = CNNEncoder()
        # 用于所有类别的全局分类的分类器层
        # Classifier layer for global classification across all classes
        self.classifier = nn.Linear(32, num_classes) # 特征维度为32 (Feature dimension is 32)

    def forward(self, x):
        feats = self.encoder(x)  # 获取特征 (Get features)
        logits = self.classifier(feats)  # 通过分类器得到 logits (Get logits from classifier)
        return logits