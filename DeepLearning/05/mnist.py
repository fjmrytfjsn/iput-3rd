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

# ここまではmnist_test.py と同じ

num_epochs = 1  # 初期設定であればMNISTは1で十分

nummax=20
x=range(200) # x軸の配列
y=np.zeros((10,200)) # y軸の配列を10本分
fig,ax = plt.subplots() # 1つのサブグラフを設定しfigとaxを取得
ax2=ax.twinx() # y2軸（右の軸）を追加

line=[0]*10 # 折れ線オブジェクトの保存用
for i in range(10):
    line[i],=ax.plot(x,y[i],label=str(i))  # ダミーの折れ線グラフを描いて折れ線オブジェクトlineの取得
line2,=ax2.plot(x,y[0],label="Acc",color="red")  # 精度のグラフの折れ線オブジェクトline2の取得

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # バッチサイズを1に設定

trained=0
for epoch in range(num_epochs):
    # 学習
    model.train()  # modelを学習モードにする
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        trained+=len(data)

        if(batch_idx%16==0):
            print("epoch",epoch,"/",num_epochs,trained,"/",len(train_dataset)) # 途中経過の表示
            # 評価
            model.eval()  # modelを評価モードにする
            total=0
            correct=0
            with torch.no_grad():  # 勾配計算をオフにする
                for n in range(10):  # 0を20個、1を20個、、、の順で200個評価する
                    num=0
                    for data, target in test_loader:
                        if(target[0]==n):  # 評価したい数字でなければ飛ばす
                            data, target = data.to(device), target.to(device)  # GPUに転送
                            output = model(data)  # 実際の実行
                            _, predicted = torch.max(output.data, 1)  # 出力は全数字の評価値なので、最大値を取る
                            total += target.size(0)  # 1のはずだけど念のため
                            correct += (predicted == target).sum().item() # 正解数を保存

                            for i in range(10): # 0-9の評価値をすべて保存
                                y[i][n*20+num]=output[0][i]
                            num+=1
                            if(num==20):  # 20個評価したので次へ
                                break
                print(correct,"/",total)
                cor=np.full(200,correct/total) # 正解率=精度 （水平線になるように200点並べる）

                ymax=max(list(map(lambda x: max(x), y))) # y軸調整用に最大値(2次元)を取得
                ymin=min(list(map(lambda x: min(x), y))) # 最小値
                for n in range(10):
                    line[n].set_ydata(y[n]) # 10本の折れ線グラフを更新

                ax.set_ylim(ymin-1,ymax+1) # y軸調整
                line2.set_ydata(cor) # 正解率を水平線で表示
                ax2.set_ylim(0,1.01) # 正解率は[0,1]
                lines_1, labels_1 = ax.get_legend_handles_labels() # 左y軸
                lines_2, labels_2 = ax2.get_legend_handles_labels() # 右y軸
                lines = lines_1 + lines_2  # 1つのグラフになるようにまとめる
                labels = labels_1 + labels_2
                ax.legend(lines, labels,loc="lower center",ncol=5) # 左右y軸を凡例に表示する
                # fig.canvas.draw() # グラフ更新
                fig.show()  # pyCharm用、VS codeだと表示が見えないかも

    print('Epoch: {}, Loss: {:.6f}'.format(epoch + 1, loss.item()))

# batch_sizeを1にしていたので戻す
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 評価 (testを全部評価する)
model.eval()  # modelを評価モードにする
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print('Accuracy on the test dataset: {:.2f}%'.format(accuracy))
