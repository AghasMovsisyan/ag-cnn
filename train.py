import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import tqdm
import os
import glob

from src.dataset import RadioDataset
from src.ag_cnn import AG_CNN
from utils.metrics import evaluate_and_plot

DATA_DIR = "data/train"
BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_names = ["Car", "Human", "Noise"]
num_classes = len(class_names)

os.makedirs("models", exist_ok=True)

full_dataset = RadioDataset(DATA_DIR, train=True)
total_size = len(full_dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print("Dataset sizes:", train_size, val_size, test_size)

g = torch.Generator().manual_seed(42)
train_ds, val_ds, _ = random_split(
    full_dataset, [train_size, val_size, test_size], generator=g
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = AG_CNN(in_channels=3, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

existing_runs = glob.glob("models/run_*")
run_id = len(existing_runs)
run_dir = f"models/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
print(f">>> Training session folder: {run_dir}")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    print(
        f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
    )

    model.eval()
    all_preds, all_labels = [], []
    val_running_loss = 0.0

    val_plots_dir = os.path.join(run_dir, "validation_plots")
    os.makedirs(val_plots_dir, exist_ok=True)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_running_loss += loss.item()

            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_acc = (
        (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    )

    print(
        f"Epoch {epoch+1} | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}"
    )

    evaluate_and_plot(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        save_dir=val_plots_dir,
        title_prefix=f"Epoch {epoch+1} Validation",
    )

    epoch_model_path = os.path.join(run_dir, f"model_epoch{epoch}.pth")
    torch.save(model.state_dict(), epoch_model_path)
    print(f">>> Saved epoch model: {epoch_model_path}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(run_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(">>> Best model updated")

print(f">>> Training finished. Best model saved at: {best_model_path}")
