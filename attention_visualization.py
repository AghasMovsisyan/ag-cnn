# attention_visualization.py
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

from dataset import RadioDataset  # make sure this works
from ag_cnn import AG_CNN         # your model file

def save_attention_samples(dataset, model, output_root="attention_dataset", device="cpu", num_samples=6000, seed=42):
    """
    Go through a sample of images in the dataset, save:
      - Original image
      - Attention heatmaps
      - Overlay images
    Each sample gets its own folder inside `output_root`.
    """
    os.makedirs(output_root, exist_ok=True)
    model.eval()

    total_samples = min(num_samples, len(dataset))
    random.seed(seed)
    sample_indices = random.sample(range(len(dataset)), total_samples)
    
    with torch.no_grad():
        for idx_count, idx in enumerate(sample_indices):
            img, _ = dataset[idx]  # ignore label
            img = img.unsqueeze(0).to(device)
            
            logits, att_maps = model(img, return_att=True)
            
            img_folder = os.path.join(output_root, f"sample_{idx_count:03d}")
            os.makedirs(img_folder, exist_ok=True)
            
            input_np = img[0].permute(1,2,0).cpu().numpy()
            input_np_norm = (input_np - input_np.min()) / (input_np.max() - input_np.min())
            plt.imsave(os.path.join(img_folder, "original.png"), input_np_norm)
            
            for i, att in enumerate(att_maps):
                att_np = att[0,0].cpu().numpy()
                att_resized = np.array(Image.fromarray(att_np*255).resize(
                    (input_np.shape[1], input_np.shape[0])
                )) / 255.0
                
                # Heatmap
                plt.imsave(os.path.join(img_folder, f"att_map_{i+1}.png"), att_resized, cmap='jet')
                
                # Overlay
                overlay = (0.5 * input_np_norm + 0.5 * plt.cm.jet(att_resized)[:,:,:3])
                plt.imsave(os.path.join(img_folder, f"overlay_{i+1}.png"), overlay)
            
            print(f"> Saved attention maps for sample {idx_count} in {img_folder}")

# -------------------------
# Run script directly
# -------------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RadioDataset("data/train", train=True)
    model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))

    # Run the attention visualization
    save_attention_samples(dataset, model, output_root="attention_samples", device=DEVICE, num_samples=20)
