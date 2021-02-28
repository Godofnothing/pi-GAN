import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from functools import  partial

from pathlib import Path

from PIL import Image

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return T.functional.resize(image, min_size)
    return image

class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size,
        transparent = False,
        aug_prob = 0.,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()

        self.data_dir = data_dir
        self.paths = [p for ext in exts for p in Path(f'{data_dir}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {data_dir} for training'
        self.set_transforms(image_size)

    def set_transforms(self, image_size):
        self.transforms = T.Compose([
            T.Lambda(partial(resize_to_minimum_size, image_size)),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transforms(img)

class ImageLoader(pl.LightningDataModule):
 
    def __init__(
        self, 
        image_dataset, 
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super(ImageLoader, self).__init__()

        self.image_dataset = image_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.image_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle = True, 
            drop_last = True
        )

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass