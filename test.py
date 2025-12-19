import os
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import RadioDataset
from src.ag_cnn import AG_CNN
from utils.metrics import evaluate_and_plot


DATA_DIR = "data/train"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Car", "Human", "Noise"]


full_dataset = RadioDataset(DATA_DIR, train=True)
total_size = len(full_dataset)

train_size = int(total_size * 0.70)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size

g = torch.Generator().manual_seed(42)
_, _, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size], generator=g
)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model_path = "models/run_15/best_model.pth"
assert os.path.exists(model_path), f"Model path does not exist: {model_path}"

model = AG_CNN(in_channels=3, num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f">>> Loaded best model from {model_path}")

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_plots_dir = "models/run_15/test_plots"
os.makedirs(test_plots_dir, exist_ok=True)

evaluate_and_plot(
    y_true=all_labels,
    y_pred=all_preds,
    class_names=CLASS_NAMES,
    save_dir=test_plots_dir,
    title_prefix="Test",
)

test_acc = sum(p == y for p, y in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nFinal TEST Accuracy = {test_acc:.4f}")
