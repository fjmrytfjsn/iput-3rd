import torch
from model import SimpleCNN
from train_with_time import train_model
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def run_experiments():
    # 前処理
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.ImageFolder("train", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータ設定
    num_epochs = 5
    activation_list = ["relu"]

    results = {}  # 結果を格納

    for act in activation_list:
        print(f"==== 実験: 活性化関数={act} ====")
        model = SimpleCNN(num_classes=2, activation=act)
        model = SimpleCNN(num_classes=2, activation=act)
        train_losses, val_accuracies, epoch_times = train_model(
            model,
            train_loader,
            val_loader,
            device,
            num_epochs=num_epochs,
            learning_rate=0.0005,
        )
        results[act] = {
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "epoch_times": epoch_times,
        }
    return results
