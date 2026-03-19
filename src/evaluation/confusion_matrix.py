import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.neu_dataset import NEUDataset
from src.models.classifier import build_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, image_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to validation images dir")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_png", type=str, default="outputs/figures/confusion_matrix_resnet18.png",
                        help="Path to save confusion matrix figure")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, class_names, image_size = load_checkpoint(args.checkpoint, device)
    transform = build_transform(image_size)

    dataset = NEUDataset(
        root_dir=args.val_dir,
        transform=transform,
        class_names=class_names
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)

    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30, values_format="d")
    plt.title("Confusion Matrix - ResNet18 on NEU-DET Validation Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to: {output_path}")


if __name__ == "__main__":
    main()