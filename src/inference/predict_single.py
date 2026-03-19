import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

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


def predict_image(model, image_path: str, transform, class_names, device, top_k: int = 3):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    top_probs = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        top_predictions.append({
            "class_name": class_names[idx],
            "probability": round(prob, 4)
        })

    predicted_class = top_predictions[0]["class_name"]
    confidence = top_predictions[0]["probability"]

    return predicted_class, confidence, top_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-k predictions to show")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, class_names, image_size = load_checkpoint(args.checkpoint, device)
    transform = build_transform(image_size)

    predicted_class, confidence, top_predictions = predict_image(
        model=model,
        image_path=args.image_path,
        transform=transform,
        class_names=class_names,
        device=device,
        top_k=args.top_k
    )

    print("\n=== Prediction Result ===")
    print(f"Image: {args.image_path}")
    print(f"Predicted defect: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    print("\nTop predictions:")
    for i, pred in enumerate(top_predictions, start=1):
        print(f"{i}. {pred['class_name']} -> {pred['probability']:.4f}")


if __name__ == "__main__":
    main()