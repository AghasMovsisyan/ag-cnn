import os
import torch
from src.ag_cnn import AG_CNN
from src.dataset import RadioDataset
from utils.attention_visualization import save_attention_samples_by_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = RadioDataset("data/train", train=True)

    model_path = "models/run_20/best_model.pth"

    run_name = os.path.basename(os.path.dirname(model_path))
    output_root = os.path.join("attention_samples", run_name)

    model = AG_CNN(in_channels=3, num_classes=3).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    save_attention_samples_by_class(
        dataset=dataset,
        model=model,
        class_names=["Car", "Human", "Noise"],
        output_root=output_root,
        device=device,
        num_samples=6000,
    )



if __name__ == "__main__":
    main()
