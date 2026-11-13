"""Image loading: aspect-preserving letterbox, augmentation, dataset class."""
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .config import AUGMENTATION, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class Letterbox:
    """Resize keeping the aspect ratio, then pad with black to size x size."""

    def __init__(self, size=IMAGE_SIZE):
        self.size = size

    def __call__(self, image):
        image = ImageOps.contain(image, (self.size, self.size), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


def build_transforms(augmentation="none"):
    """Return (train_transform, eval_transform)."""
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_ops = [Letterbox()]
    if augmentation != "none":
        train_ops.append(transforms.RandomAffine(**AUGMENTATION[augmentation]))
    return transforms.Compose(train_ops + norm), transforms.Compose([Letterbox()] + norm)


class UltrasoundDataset(Dataset):
    """Rows need the columns `image_path` and `label` (int)."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def load_split(split_dir):
    """Read split_dev.csv / split_test.csv written by scripts/make_splits.py."""
    from pathlib import Path
    d = Path(split_dir)
    return pd.read_csv(d / "split_dev.csv"), pd.read_csv(d / "split_test.csv")
