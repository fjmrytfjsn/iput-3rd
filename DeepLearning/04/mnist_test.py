import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchinfo
import numpy as np
from matplotlib import pyplot as plt

# ハイパーパラメータ設定
batch_size = 128
# learning_rate = 0.01  # SGD用

# データセットのダウンロードと前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    # (0,1)の範囲で黒(0)の領域の方が多いため、(-1,1)の範囲で平均が0になるようにする
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNISTデータセットの読み込み、初回は自動でダウンロードする。
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
print("train=",len(train_dataset))  # データ数を確認
print("test=",len(test_dataset))    # データ数を確認
print("type(train_dataset)=",type(train_dataset))
print("type(train_dataset[0])=",type(train_dataset[0]))
print("type(train_dataset[0][0])=",type(train_dataset[0][0]))
print("train_dataset[0][0].shape=",train_dataset[0][0].shape)
print("type(train_dataset[0][1])=",type(train_dataset[0][1]))
#print(train_dataset.shape,train_dataset[0])

# データローダを作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)  # 畳み込み層1
        self.conv2 = nn.Conv2d(4, 4, 3)  # 畳み込み層2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4*24*24, 128)       # 全結合層
        self.fc2 = nn.Linear(128, 10)  # 全結合層

    def forward(self, xx):
        xx = self.conv1(xx)  # 1x28x28->4x26x26
        xx = nn.ReLU()(xx)
        xx = self.conv2(xx)  # 4x26x26->4x24x24
        xx = nn.ReLU()(xx)
        xx = self.flatten(xx)  # 4x24x24->2304
        xx = self.fc1(xx)  # 4x24x24->2304
        xx = nn.ReLU()(xx)
        xx = self.fc2(xx)
        return xx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MyModel().to(device)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters())

print(train_dataset[0][0].shape)
torchinfo.summary(model=model, input_size=(1, 1,28,28))
