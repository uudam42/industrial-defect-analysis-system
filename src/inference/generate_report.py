import argparse
import json
from pathlib import Path

from src.inference.predict_single import (
    get_device,
    load_checkpoint,
    build_transform,
    predict_image,
)


DEFECT_KNOWLEDGE = {
    "crazing": {
        "severity": "medium",
        "visual_features": [
            "irregular crack-like lines",
            "surface fracture patterns"
        ],
        "description": "The image likely contains crazing defects characterized by crack-like surface lines.",
        "recommended_action": "Inspect crack density and depth, and determine whether the sample should be rejected."
    },
    "inclusion": {
        "severity": "medium",
        "visual_features": [
            "embedded foreign material",
            "localized irregular texture"
        ],
        "description": "The image likely contains inclusion defects caused by foreign material or internal impurities.",
        "recommended_action": "Perform material review and inspect whether the defect affects downstream processing."
    },
    "patches": {
        "severity": "medium",
        "visual_features": [
            "patch-like regions",
            "localized surface inconsistency"
        ],
        "description": "The image likely contains patch defects with visible local inconsistency on the surface.",
        "recommended_action": "Verify patch area and assess whether the surface can be reworked."
    },
    "pitted_surface": {
        "severity": "high",
        "visual_features": [
            "small pit-like marks",
            "surface depressions"
        ],
        "description": "The image likely contains pitted surface defects with visible depressions and texture damage.",
        "recommended_action": "Inspect pit density and depth; consider rejection if pits are severe."
    },
    "rolled-in_scale": {
        "severity": "high",
        "visual_features": [
            "scale-like embedded marks",
            "irregular elongated surface contamination"
        ],
        "description": "The image likely contains rolled-in scale defects caused by oxide scale being pressed into the surface.",
        "recommended_action": "Check severity and extent of embedded scale before deciding on rework or rejection."
    },
    "scratches": {
        "severity": "medium",
        "visual_features": [
            "linear marks",
            "surface texture disruption"
        ],
        "description": "The image is highly likely to contain scratch-type surface defects.",
        "recommended_action": "Perform manual inspection and verify whether polishing or rework is sufficient."
    },
}


def build_report(predicted_class: str, confidence: float, image_path: str, top_predictions):
    defect_info = DEFECT_KNOWLEDGE.get(predicted_class, {
        "severity": "unknown",
        "visual_features": [],
        "description": "No template description available.",
        "recommended_action": "Manual inspection is recommended."
    })

    report = {
        "image_path": image_path,
        "defect_type": predicted_class,
        "confidence": round(confidence, 4),
        "severity": defect_info["severity"],
        "visual_features": defect_info["visual_features"],
        "description": defect_info["description"],
        "recommended_action": defect_info["recommended_action"],
        "top_predictions": top_predictions,
    }

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output_json", type=str, default="outputs/reports/prediction_report.json",
                        help="Path to save generated JSON report")
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
        top_k=3
    )

    report = build_report(
        predicted_class=predicted_class,
        confidence=confidence,
        image_path=args.image_path,
        top_predictions=top_predictions
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Structured Defect Report ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()