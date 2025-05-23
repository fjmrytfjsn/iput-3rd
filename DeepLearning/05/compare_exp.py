import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

# データ: [総学習時間, 精度, ラベル]
data = [
    [280, 0.5014, "lr=0.01, bs=32, relu"],
    [284, 0.4986, "lr=0.01, bs=32, sigmoid"],
    [279, 0.5014, "lr=0.005, bs=32, relu"],
    [300, 0.5014, "lr=0.005, bs=32, sigmoid"],
    [297, 0.7880, "lr=0.0001, bs=32, relu"],
    [332, 0.4976, "lr=0.0001, bs=32, sigmoid"],
    [278, 0.7884, "lr=0.00005, bs=32, relu"],
    [282, 0.4914, "lr=0.00005, bs=32, sigmoid"],
    [255, 0.7928, "lr=0.00005, bs=64, relu"],
]

fig, ax = plt.subplots(figsize=(8, 6))
for t, acc, label in data:
    plt.scatter(t, acc, label=label)

plt.xlabel("総学習時間 [秒]")
plt.ylabel("エポック5時点の検証精度")
plt.title("総学習時間と精度の関係（エポック5）")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
