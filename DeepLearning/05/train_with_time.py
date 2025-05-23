import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


def train_model(
    model, train_loader, val_loader, device, num_epochs=3, learning_rate=0.001
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []
    epoch_times = []

    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_accuracy:.4f}, Time={epoch_time:.2f}s"
        )

    return train_losses, val_accuracies, epoch_times
