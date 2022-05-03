from torch.utils.data import Dataset
from PIL import Image

import glob
import os
from typing import Optional, Callable


class HumansMattingDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.img_path = os.path.join(root, "clip_img")
        self.mask_path = os.path.join(root, "matting")
        self.transform = transform
        self.target_transform = target_transform

        self.img_list = self.get_filenames(self.img_path, ".jpg")
        self.mask_list = self.get_filenames(self.mask_path, ".png")
        self._filter_mismatched()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])
        return self.transform(img), self.target_transform(mask)

    def get_filenames(self, path, extension):
        files_list = list()
        for filename in glob.iglob(path + '/**/*' + extension, recursive=True):
            files_list.append(filename)
        return files_list

    def _filter_mismatched(self): # there are some extra masks
        for img_path, mask_path in zip(self.img_list, self.mask_list):
            if img_path.split("/")[-1].split(".")[0] != mask_path.split("/")[-1].split(".")[0]:
                self.mask_list.remove(mask_path)
