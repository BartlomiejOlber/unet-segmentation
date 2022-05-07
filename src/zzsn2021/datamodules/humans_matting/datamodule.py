import os
from typing import Any, Callable, List, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.zzsn2021.datasets.humans_matting.dataset import HumansMattingDataset


class HumansMattingDataModule(LightningDataModule):

    name: str = "humans_matting"
    #: Dataset class to use
    dataset_cls = HumansMattingDataset

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(*args, **kwargs)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        print(self.num_workers)
        print(self.batch_size)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        # default_transforms = self.default_transforms()
        # self.dataset_cls(self.data_dir, transform=default_transforms[0], target_transform=default_transforms[1])


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """
        print("SETUP")
        if stage == "fit" or stage is None:
            train_transforms = self.resize_transforms() if self.train_transforms is None else self.train_transforms
            # val_transforms = self.default_test_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(self.data_dir, transform=train_transforms, target_transform=train_transforms)
            # dataset_val = self.dataset_cls(self.data_dir, transform=train_transforms, target_transform=train_transforms) #todo eager loading

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_train, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.resize_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, transform=test_transforms[0], target_transform=test_transforms[1]
            )

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """
        Splits the dataset into train and validation set
        """
        len_dataset = len(dataset)  # type: ignore[arg-type]
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(self.seed))

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """
        Computes split lengths for train and validation set
        """
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f'Unsupported type {type(self.val_split)}')

        return splits

    def default_transforms(self) -> Callable: #todo experiments, imgs augmentation, resize
        img_transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(64)
        ])
        target_transforms = transforms.Compose([transforms.ToTensor(), ExtractAlpha(), transforms.Resize(64)])
        return img_transforms, target_transforms

    def resize_transforms(self) -> Callable: #todo experiments, imgs augmentation, resize
        img_transforms = transforms.Compose([
            # transforms.Resize(64)
            torch.nn.Identity()
        ])
        return img_transforms

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


class ExtractAlpha(object):
    def __call__(self, pic):
        return pic[3, :, :].unsqueeze(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'