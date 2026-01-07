import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


def save_attention_samples_by_class(
    dataset,
    model,
    class_names,
    output_root="attention_dataset",
    device="cpu",
    num_samples=6000,
    seed=42,
):
    """
    Save:
      - original image
      - attention heatmaps
      - overlay images
    Folder naming: real_<true_class>_pred_<pred_class>
    """

    os.makedirs(output_root, exist_ok=True)
    model.eval()

    total_samples = min(num_samples, len(dataset))
    random.seed(seed)
    sample_indices = random.sample(range(len(dataset)), total_samples)

    with torch.no_grad():
        for idx_count, idx in enumerate(sample_indices):
            img, label = dataset[idx]  # label = true class index
            img_batch = img.unsqueeze(0).to(device)

            logits, att_maps = model(img_batch, return_att=True)
            pred_idx = logits.argmax(1).item()

            true_class = class_names[label]
            pred_class = class_names[pred_idx]

            sample_dir = os.path.join(
                output_root, f"real_{true_class}_pred_{pred_class}_{idx_count:04d}"
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Save original image
            input_np = img.permute(1, 2, 0).cpu().numpy()
            input_np = (input_np - input_np.min()) / (
                input_np.max() - input_np.min() + 1e-8
            )
            plt.imsave(os.path.join(sample_dir, "original.png"), input_np)

            # Save attention maps and overlays
            for i, att in enumerate(att_maps):
                att_np = att[0, 0].cpu().numpy()
                att_resized = (
                    np.array(
                        Image.fromarray((att_np * 255).astype(np.uint8)).resize(
                            (input_np.shape[1], input_np.shape[0])
                        )
                    )
                    / 255.0
                )

                plt.imsave(
                    os.path.join(sample_dir, f"att_map_{i+1}.png"),
                    att_resized,
                    cmap="jet",
                )
                overlay = 0.5 * input_np + 0.5 * plt.cm.jet(att_resized)[:, :, :3]
                plt.imsave(os.path.join(sample_dir, f"overlay_{i+1}.png"), overlay)

            print(f"> Saved sample {idx_count}: {true_class} -> {pred_class}")
