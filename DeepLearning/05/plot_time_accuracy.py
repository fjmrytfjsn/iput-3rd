import matplotlib.pyplot as plt
from experiment import run_experiments

plt.rcParams["font.family"] = "Meiryo"


def plot_time_and_accuracy_all(results):
    plt.figure(figsize=(12, 6))

    # 学習時間の推移
    plt.subplot(1, 2, 1)
    for act, res in results.items():
        plt.plot(
            range(1, len(res["epoch_times"]) + 1),
            res["epoch_times"],
            marker="o",
            label=f"{act} 学習時間",
        )
    plt.xlabel("エポック数")
    plt.ylabel("学習時間（秒）")
    plt.title("各エポックの学習時間")
    plt.legend()

    # 検証精度の推移
    plt.subplot(1, 2, 2)
    for act, res in results.items():
        plt.plot(
            range(1, len(res["val_accuracies"]) + 1),
            res["val_accuracies"],
            marker="x",
            label=f"{act} 精度",
        )
    plt.xlabel("エポック数")
    plt.ylabel("検証精度")
    plt.title("各エポックの検証精度")
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("time_accuracy_batch_size_64.png")


if __name__ == "__main__":
    results = run_experiments()
    plot_time_and_accuracy_all(results)
