import torch
import torch.nn as nn
import torch.nn.functional as F

class KANHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(KANHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)  # 第一层全连接
        self.fc2 = nn.Linear(512, 256)           # 第二层全连接
        self.fc3 = nn.Linear(256, num_classes)  # 输出层
        self.dropout = nn.Dropout(0.5)           # Dropout层

    def forward(self, x):
        # print("Output shape:", x.shape)
        x = x.view(x.size(0), -1)  # 调整维度
        # print("Output shape:", x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class BiTWithKAN(nn.Module):
    def __init__(self, bit_model, kan_head):
        super(BiTWithKAN, self).__init__()
        self.bit_model = bit_model
        self.kan_head = kan_head

    def forward(self, x):
        features = self.bit_model(x)  # 提取特征
        output = self.kan_head(features)  # 通过分类头
        return output
