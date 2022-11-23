"""Sub package containing the dataset loader class"""

from typing import Callable, Tuple

import numpy as np
from torch.utils.data import Dataset

from ._logger import logger


class DatasetLoader(Dataset):
    """Dataset Loader Class"""

    def __init__(self, images: np.ndarray,
                 densities: np.ndarray,
                 transform: Callable = None):
        self.transform = transform

        if images.ndim == 3:
            new_shape = (images.shape[0], 1, *images.shape[1:])
            logger.info('Input image dim is 3. Assuming input image is gray scale. '
                        f'Adjusting shape from {images.shape} to {new_shape}')
            images = images.reshape((images.shape[0], 1, *images.shape[1:]))
        self.images = images
        # self.images = (images - np.min(images)) / (np.max(images) - np.min(images))

        if densities.ndim == 3:
            new_shape = (densities.shape[0], 1, *densities.shape[1:])
            logger.info('Density (label) array has dimension 3 but must be 4. '
                        f'Adjusting shape from {densities.shape} to {new_shape}')
            densities = densities.reshape((densities.shape[0], 1, *densities.shape[1:]))
        assert images.shape[2:] == densities.shape[2:]
        assert images.ndim == densities.ndim
        self.densities = densities  # normalize densities to 0-1 so that the sum is really the number of objects

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item) -> Tuple[np.ndarray]:
        images = self.images[item, ...]
        densities = self.densities[item, ...]
        if self.transform is not None:
            augmentations = self.transform(images, densities)
            images = augmentations["image"]
            densities = augmentations["mask"]

        return images, densities
