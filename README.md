# Industrial Defect Analysis System

A lightweight industrial surface defect classification and structured analysis system built with PyTorch on the NEU-DET dataset.

This project starts from a classification baseline and extends it into a practical inspection-oriented pipeline with:
- defect classification
- model comparison
- confusion matrix visualization
- single-image inference
- JSON-formatted structured defect reports

---

## 1. Project Overview

#### Results Preview

- ResNet18 baseline validation accuracy: **96.67%**
- EfficientNet-B0 validation accuracy: **100.00%**


Industrial surface defect inspection is an important task in quality control for manufacturing and steel processing.  
This project focuses on classifying common steel surface defects from images and generating structured inspection outputs for practical use.

The system supports:
- training defect classifiers on NEU-DET
- comparing different CNN backbones
- running inference on single images
- generating structured JSON reports
- visualizing model performance with confusion matrices

---

## 2. Dataset

This project uses the **NEU-DET** industrial defect dataset.

Defect classes used in this project::
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Dataset structure used in this project:
> Note: The raw NEU-DET dataset is not included in this repository.  
> Please download the dataset separately and place it under `data/raw/NEU-DET/` before running training or inference scripts.

```text
data/raw/NEU-DET/
├── train/
│   ├── annotations/
│   └── images/
│       ├── crazing/
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/
    ├── annotations/
    └── images/
        ├── crazing/
        ├── inclusion/
        ├── patches/
        ├── pitted_surface/
        ├── rolled-in_scale/
        └── scratches/
```

## 3. Project Structure
```text
IndustDefect_VLM/
├── data/
│   └── raw/
│       └── NEU-DET/
├── outputs/
│   ├── figures/
│   ├── models/
│   │   └── baselines/
│   └── reports/
├── src/
│   ├── datasets/
│   │   └── neu_dataset.py
│   ├── evaluation/
│   │   └── confusion_matrix.py
│   ├── inference/
│   │   ├── generate_report.py
│   │   └── predict_single.py
│   ├── models/
│   │   └── classifier.py
│   └── training/
│       └── train.py
├── requirements.txt
└── README.md
```

## 4. Methods

#### 4.1 Baseline Model

A ResNet18 classifier was used as the first baseline model for six-class industrial defect classification.

#### 4.2 Improved Model

An EfficientNet-B0 classifier was trained under the same setup for comparison.

#### 4.3 Structured Analysis

The project extends pure classification into a lightweight analysis pipeline by generating JSON-formatted inspection reports containing:

- predicted defect type
- confidence
- severity
- visual features
- recommended action

## 5. Experimental Results
#### 5.1 Model Comparison
```markdown
| Model           | Image Size | Batch Size | Validation Accuracy | Macro F1 |
| --------------- | ---------- | ---------- | ------------------- | -------- |
| ResNet18        | 224        | 4          | 0.9667              | 0.9666   |
| EfficientNet-B0 | 224        | 4          | 1.0000              | 1.0000   |
```

#### 5.2 ResNet18 Analysis

ResNet18 achieved strong baseline performance, with the main confusion occurring between:

- inclusion
- pitted_surface

#### 5.3 EfficientNet-B0 Analysis

EfficientNet-B0 outperformed ResNet18 on the current validation split and achieved perfect classification on the validation set used in this project.

## 6. Confusion Matrices
ResNet18

<img width="2400" height="2400" alt="confusion_matrix_resnet18" src="https://github.com/user-attachments/assets/bf232acb-2844-445d-845e-269a9915938e" />

Saved at:
```
outputs/figures/confusion_matrix_resnet18.png
```

EfficientNet-B0

<img width="2400" height="2400" alt="confusion_matrix_efficientnet_b0" src="https://github.com/user-attachments/assets/b92893e7-c91f-4c6f-9f37-bb22ddf3526f" />

Saved at:
```
outputs/figures/confusion_matrix_efficientnet_b0.png
```

## 7. Single-Image Inference

Run prediction on one image:
```
python -m src.inference.predict_single \
  --image_path data/raw/NEU-DET/validation/images/scratches/scratches_249.jpg \
  --checkpoint outputs/models/baselines/resnet18_v1/best_resnet18.pth
```
Example output:

```
=== Prediction Result ===
Image: data/raw/NEU-DET/validation/images/scratches/scratches_249.jpg
Predicted defect: scratches
Confidence: 1.0000

Top predictions:
1. scratches -> 1.0000
2. crazing -> 0.0000
3. pitted_surface -> 0.0000
```

## 8. Structured Report Generation
Generate a JSON report for one image:
```
python -m src.inference.generate_report \
  --image_path data/raw/NEU-DET/validation/images/inclusion/inclusion_299.jpg \
  --checkpoint outputs/models/baselines/resnet18_v1/best_resnet18.pth \
  --output_json outputs/reports/inclusion_299_report.json
```

Example JSON output:
```
{
  "image_path": "data/raw/NEU-DET/validation/images/inclusion/inclusion_299.jpg",
  "defect_type": "inclusion",
  "confidence": 0.989,
  "severity": "medium",
  "visual_features": [
    "embedded foreign material",
    "localized irregular texture"
  ],
  "description": "The image likely contains inclusion defects caused by foreign material or internal impurities.",
  "recommended_action": "Perform material review and inspect whether the defect affects downstream processing.",
  "top_predictions": [
    {
      "class_name": "inclusion",
      "probability": 0.989
    },
    {
      "class_name": "pitted_surface",
      "probability": 0.0048
    },
    {
      "class_name": "scratches",
      "probability": 0.0048
    }
  ]
}
```

## 9. Example Structured Reports

Generated report examples:
```
outputs/reports/scratches_249_report.json

outputs/reports/inclusion_299_report.json

outputs/reports/patches_241_report.json
```
### Example 1: Scratches Report
<img width="902" height="1357" alt="scratches_249_report" src="https://github.com/user-attachments/assets/7e07f5f2-6742-40d2-86ed-464512ba1e45" />

### Example 2: Inclusion Report
<img width="983" height="831" alt="inclusion_299_report" src="https://github.com/user-attachments/assets/4d7f20f6-9032-4669-8779-25a3af1356c1" />

### Example 3: Patches Report
<img width="896" height="882" alt="patches_241_report" src="https://github.com/user-attachments/assets/9040205e-2dd6-4bff-8643-3b64f5706b93" />


These examples show how the classifier output is converted into a structured inspection-oriented report for practical use.

## 10. Training
#### Train ResNet18
```
python -m src.training.train \
  --train_dir data/raw/NEU-DET/train/images \
  --val_dir data/raw/NEU-DET/validation/images \
  --model_name resnet18 \
  --epochs 10 \
  --batch_size 4 \
  --image_size 224
 ```

#### Train EfficientNet-B0
```
python -m src.training.train \
  --train_dir data/raw/NEU-DET/train/images \
  --val_dir data/raw/NEU-DET/validation/images \
  --model_name efficientnet_b0 \
  --epochs 10 \
  --batch_size 4 \
  --image_size 224
```

## 11. Evaluation
Generate Confusion Matrix for ResNet18
```
python -m src.evaluation.confusion_matrix \
  --val_dir data/raw/NEU-DET/validation/images \
  --checkpoint outputs/models/baselines/resnet18_v1/best_resnet18.pth \
  --output_png outputs/figures/confusion_matrix_resnet18.png
```
Generate Confusion Matrix for EfficientNet-B0
```
python -m src.evaluation.confusion_matrix \
  --val_dir data/raw/NEU-DET/validation/images \
  --checkpoint outputs/models/baselines/efficientnet_b0_v1/best_efficientnet_b0.pth \
  --output_png outputs/figures/confusion_matrix_efficientnet_b0.png
```
## 12. Baseline Records
### ResNet18 Baseline

Saved under:
```
outputs/models/baselines/resnet18_v1/
```
Files:
```
best_resnet18.pth

metrics_resnet18.json

resnet18_baseline_v1.json

resnet18_classification_report.txt

resnet18_train_config.txt
```

### EfficientNet-B0 Baseline

Saved under:
```
outputs/models/baselines/efficientnet_b0_v1/
```
Files:
```

best_efficientnet_b0.pth

metrics_efficientnet_b0.json

efficientnet_b0_baseline_v1.json

efficientnet_b0_classification_report.txt

efficientnet_b0_train_config.txt
```

## 13. Environment

Recommended setup:
- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Pillow

Install dependencies:
```
pip install -r requirements.txt
```

## 14. Key Findings

- ResNet18 provided a strong baseline with 96.67% validation accuracy.
- The main ResNet18 confusion occurred between inclusion and pitted_surface.
- EfficientNet-B0 improved the performance substantially and achieved 100% validation accuracy on the validation split used in this project.
- The project was extended from pure classification to an inspection-oriented system by generating JSON-formatted structured defect reports.

## 15. Future Improvements

Possible next steps:

- add inference latency benchmarking
- compare more lightweight backbones
- support batch report generation
- add retrieval-based similar-case search
- export deployment-ready inference pipeline

## 16. Author Notes

This project was developed as a compact but practical industrial computer vision system, with emphasis on:

- reproducible baselines
- model comparison
- interpretable outputs
- engineering-oriented project organization

It is suitable as a portfolio project for applications in:

- computer vision
- multimodal systems
- applied machine learning
- AI engineering
- industrial AI inspection
