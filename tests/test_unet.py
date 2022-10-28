import os
import pathlib
import unittest
import zipfile
from typing import List, Tuple

import h5py
import numpy as np
import wget
from PIL import Image
from scipy.ndimage import gaussian_filter

import unet
from unet.dataset import DatasetLoader

__this_dir__ = pathlib.Path(__file__).parent


def get_and_unzip(url: str, location: str = "."):
    """
    Extract a ZIP archive from given URL.
    Args:
        url: url of a ZIP file
        location: target location to extract archive in

    Taken from https://github.com/NeuroSYS-pl/objects_counting_dmap/blob/7d91bb865fe00f1c8eca17bcdac693162a981c77/get_data.py#L105
    """
    dataset = wget.download(url)
    dataset = zipfile.ZipFile(dataset)
    dataset.extractall(location)
    dataset.close()
    os.remove(dataset.filename)


def get_cell_data(image_size: Tuple[int] = None, force: bool = False,
                  train_valid_portions: Tuple[float, float] = (80., 20.,)):
    """get cell data which is open access
    Code is mostly copied from
    https://github.com/NeuroSYS-pl/objects_counting_dmap/blob/7d91bb865fe00f1c8eca17bcdac693162a981c77/get_data.py#L105
    """

    if np.sum(train_valid_portions) not in (1, 100):
        raise ValueError('The portions for train/valid/test data is wrong as it does not sum up to 100%!')
    if np.sum(train_valid_portions) == 100:
        train_valid_portions = tuple(np.array(train_valid_portions) / 100)

    if pathlib.Path('data/cells/train.hdf').exists() and pathlib.Path('data/cells/valid.hdf').exists() and not force:
        return

    sorted_image_list = sorted(pathlib.Path('data/cells').glob('*cell.png'))
    if len(sorted_image_list) < 200:
        get_and_unzip(
            'http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip',
            location='data/cells'
        )

    sorted_image_list = sorted(pathlib.Path('data/cells').glob('*cell.png'))
    sorted_label_list = sorted(pathlib.Path('data/cells').glob('*dots.png'))
    n_images = len(sorted_image_list)
    assert n_images == len(sorted_label_list)

    def write_to_h5(h5file: h5py.File, image_list: List, label_list: List):
        """write the hdf5 file"""
        for (i, img_path), label_path in zip(enumerate(image_list), label_list):
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            transposed_image = np.transpose(image, (2, 0, 1))
            if image_size is not None:
                transposed_image = transposed_image[:, 0:image_size[1], 0:image_size[0]]
            if 'images' not in h5file:
                h5file.create_dataset('images', shape=(len(image_list), *transposed_image.shape))

            # load an RGB image
            label = np.array(Image.open(label_path))
            # make a one-channel label array with 100 in red dots positions
            label = 100.0 * (label[:, :, 0] > 0)
            # generate a density map by applying a Gaussian filter
            label = gaussian_filter(label, sigma=(1, 1), order=0)

            if image_size is not None:
                label = label[0:image_size[1], 0:image_size[0]]

            if 'labels' not in h5file:
                h5file.create_dataset('labels', shape=(len(image_list), 1, *label.shape))

            # save data to HDF5 file
            h5file['images'][i] = transposed_image
            h5file['labels'][i, 0] = label

    ntrain = int(n_images * train_valid_portions[0])
    nvalid = int(n_images * train_valid_portions[1])

    assert np.sum((ntrain, nvalid)) <= n_images

    with h5py.File('data/cells/train.hdf', 'w') as h5:
        write_to_h5(h5, sorted_image_list[0:ntrain], sorted_label_list[0:ntrain])
    with h5py.File('data/cells/valid.hdf', 'w') as h5:
        write_to_h5(h5, sorted_image_list[ntrain:], sorted_label_list[ntrain:])


class TestUNet(unittest.TestCase):

    def test_unet(self):
        # get the test data
        image_size = None
        get_cell_data(image_size=image_size, force=True)

        with h5py.File('data/cells/train.hdf') as h5:
            ds = DatasetLoader(h5['images'][:], h5['labels'][:])
            image_size = h5['images'].shape[2:]

        assert ds[0][0].shape == (3, *image_size)
        assert ds[0][1].shape == (1, *image_size)

        cfg = unet.utils.load_hyperparameters('conf/hyperparameters.yaml')
        case = unet.Case(cfg, 'testrun')
        case.run()
