import numpy as np
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    """Dataset Loader Class"""

    def __init__(self, images: np.ndarray, density: np.ndarray, transform=None):
        self.transform = transform
        self.images = (images - np.min(images)) / (np.max(images) - np.min(images))
        self.density = density  # normalize density to 0-1 so that the sum is really the number of objects

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.transform is not None:
            raise NotImplementedError('Transform is not available yet')
        return self.images[item, ...], self.density[item, ...]


if __name__ == '__main__':
    test()
