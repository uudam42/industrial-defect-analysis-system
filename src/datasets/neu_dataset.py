from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class NEUDataset(Dataset):
    """
    Expected folder structure:
    data/raw/NEU-DET/train/images/
        crazing/
        inclusion/
        patches/
        pitted_surface/
        rolled-in_scale/
        scratches/

    or validation/images/...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Callable = None,
        class_names: List[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_names = class_names or [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        ]

        self.class_to_idx: Dict[str, int] = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        self.samples: List[Tuple[Path, int]] = self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No image samples found under: {self.root_dir}")

    def _is_image_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[Warning] Class folder not found: {class_dir}")
                continue

            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file() and self._is_image_file(file_path):
                    samples.append((file_path, self.class_to_idx[class_name]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(image_path)