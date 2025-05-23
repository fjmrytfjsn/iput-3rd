import matplotlib.pyplot as plt

def plot_results(train_losses, val_accuracies, title_suffix=""):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))

    # 学習損失のグラフ
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, marker='o')
    plt.title(f'学習損失の推移{title_suffix}')
    plt.xlabel('エポック数')
    plt.ylabel('損失')

    # 検証精度のグラフ
    plt.subplot(1,2,2)
    plt.plot(epochs, val_accuracies, marker='o')
    plt.title(f'検証精度の推移{title_suffix}')
    plt.xlabel('エポック数')
    plt.ylabel('精度')

    plt.tight_layout()
    plt.show()