import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        theta_x = self.theta(x)
        psi_g = self.psi(g)

        if psi_g.shape[2:] != theta_x.shape[2:]:
            psi_g = F.interpolate(
                psi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=False
            )

        f = self.relu(theta_x + psi_g)
        att_map = torch.sigmoid(self.phi(f))
        return x * att_map, att_map


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AG_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout_p=0.2):
        super().__init__()
        self.conv1 = conv_block(in_channels, 4)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_block(4, 8)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_block(8, 16)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_block(16, 32)

        self.att1 = AttentionGate(16, 32, inter_channels=4)
        self.att2 = AttentionGate(8, 32, inter_channels=2)
        self.att3 = AttentionGate(4, 32, inter_channels=1)

        self.fc = nn.Sequential(
            nn.Linear(16 + 8 + 4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, return_att=False):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        g = self.conv4(self.pool3(x3))

        o1, att1 = self.att1(x3, g)
        o2, att2 = self.att2(x2, g)
        o3, att3 = self.att3(x1, g)

        f1 = o1.flatten(2).sum(2)
        f2 = o2.flatten(2).sum(2)
        f3 = o3.flatten(2).sum(2)
        features = torch.cat([f1, f2, f3], dim=1)

        logits = self.fc(features)

        if return_att:
            return logits, [att1, att2, att3]
        return logits


import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random


def save_attention_samples(
    dataset,
    model,
    output_root="attention_dataset",
    device="cpu",
    num_samples=6000,
    seed=42,
):
    """
    Go through a sample of images in the dataset, save:
      - Original image
      - Attention heatmaps
      - Overlay images
    Each sample gets its own folder inside `output_root`.

    dataset: torch Dataset
    model: AG_CNN with return_att=True
    output_root: root folder to save everything
    device: "cuda" or "cpu"
    num_samples: number of samples to process
    seed: random seed for reproducibility
    """
    os.makedirs(output_root, exist_ok=True)
    model.eval()

    total_samples = min(num_samples, len(dataset))
    random.seed(seed)
    sample_indices = random.sample(range(len(dataset)), total_samples)

    with torch.no_grad():
        for idx_count, idx in enumerate(sample_indices):
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)

            logits, att_maps = model(img, return_att=True)

            img_folder = os.path.join(output_root, f"sample_{idx_count:03d}")
            os.makedirs(img_folder, exist_ok=True)

            input_np = img[0].permute(1, 2, 0).cpu().numpy()
            input_np_norm = (input_np - input_np.min()) / (
                input_np.max() - input_np.min()
            )
            orig_path = os.path.join(img_folder, "original.png")
            plt.imsave(orig_path, input_np_norm)

            for i, att in enumerate(att_maps):
                att_np = att[0, 0].cpu().numpy()
                att_resized = (
                    np.array(
                        Image.fromarray(att_np * 255).resize(
                            (input_np.shape[1], input_np.shape[0])
                        )
                    )
                    / 255.0
                )

                heatmap_path = os.path.join(img_folder, f"att_map_{i+1}.png")
                plt.imsave(heatmap_path, att_resized, cmap="jet")

                overlay = 0.5 * input_np_norm + 0.5 * plt.cm.jet(att_resized)[:, :, :3]
                overlay_path = os.path.join(img_folder, f"overlay_{i+1}.png")
                plt.imsave(overlay_path, overlay)

            print(f"> Saved attention maps for sample {idx_count} in {img_folder}")


if __name__ == "__main__":
    from dataset import RadioDataset

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RadioDataset("data/train", train=True)
    model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)
    model.load_state_dict(
        torch.load("models/run_14/best_model.pth", map_location=DEVICE)
    )

    save_attention_samples(
        dataset, model, output_root="attention_samples", device=DEVICE
    )
