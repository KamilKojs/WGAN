from typing import Optional, Callable, Tuple, Any
from pathlib import Path

import torch

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

from src.data.transforms import default_transform

class CelebDataset(VisionDataset):
    """
    Labeled dataset used for model training
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = default_transform(),
    ) -> None:
        super().__init__(root, transform=transform)

        samples = _load_samples(root)
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is ground truth of the target class.
        """
        path= self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        noise = torch.randn(100, 1, 1, device="cpu")

        return sample, noise

    def __len__(self) -> int:
        return len(self.samples)


def _load_samples(images_dir: str):
    samples = [
        sample_path
        for sample_path 
        in Path(images_dir).rglob("*.jpg")
    ]
    return samples