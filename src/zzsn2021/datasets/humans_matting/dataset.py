import pickle

from torch.utils.data import Dataset
from PIL import Image

import glob
import os
from typing import Optional, Callable
from tqdm import tqdm


class HumansMattingDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, lazy=False):
        self.root = root
        self.img_path = os.path.join(self.root, "clip_img")
        self.mask_path = os.path.join(self.root, "matting")
        self.transform = transform
        self.target_transform = target_transform

        self.img_path_list = self.get_filenames(self.img_path, ".jpg")
        self.mask_path_list = self.get_filenames(self.mask_path, ".png")
        self._filter_mismatched()
        self.lazy = lazy
        if not self.lazy:
            self._load_data()

    def _load_data(self):
        pickle_path = os.path.join(self.root, 'dataset.pkl')
        try:
            with open(pickle_path, 'rb') as f:
                self.img_list, self.mask_list = pickle.load(f)
        except IOError:
            self.img_list = []
            self.mask_list = []
            for img_path in tqdm(self.img_path_list):
                img = Image.open(img_path)
                keep = img.copy()
                self.img_list.append(self.transform(keep))
                img.close()

            for mask_path in tqdm(self.mask_path_list):
                mask = Image.open(mask_path)
                keep = mask.copy()
                self.mask_list.append(self.target_transform(keep))
                mask.close()

            with open(pickle_path, 'wb') as f:
                pickle.dump([self.img_list, self.mask_list], f)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if not self.lazy:
            return self.transform(self.img_list[idx]), self.target_transform(self.mask_list[idx])
        img = Image.open(self.img_path_list[idx])
        mask = Image.open(self.mask_path_list[idx])
        return self.transform(img), self.target_transform(mask)

    def get_filenames(self, path, extension):
        files_list = list()
        for filename in glob.iglob(path + '/**/*' + extension, recursive=True):
            files_list.append(filename)
        return sorted(files_list)

    def _filter_mismatched(self):  # there are some extra masks
        for img_path, mask_path in zip(self.img_path_list, self.mask_path_list):
            if img_path.split("/")[-1].split(".")[0] != mask_path.split("/")[-1].split(".")[0]:
                self.mask_path_list.remove(mask_path)
                break
