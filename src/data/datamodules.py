from src.data.datasets import CelebDataset
from src.data.transforms import default_transform

from torch.utils.data import DataLoader
import pytorch_lightning as pl

class CelebDataModule(pl.LightningDataModule):
    """PyTorch-Lightning data module for WGAN"""

    def __init__(
        self,
        train_dataset_dir: str,
        train_batch_size: int = 4,
        num_workers: int = 12,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        self.train_dataset_dir = train_dataset_dir
        self.dataset_transform = default_transform()
        self.train_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset = self._load_dataset(
                self.train_dataset_dir,
                self.dataset_transform,
            )

        if stage == "test" or stage is None:
            pass

    @staticmethod
    def _load_dataset(dataset_dir: str, transform):
        return CelebDataset(root=dataset_dir, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )
