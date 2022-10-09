import pathlib

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import re

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import DatasetLoader
from typing import Dict

LOADER = yaml.SafeLoader
LOADER.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

RGB_WEIGHTS = [0.2989, 0.5870, 0.1140]


def rgb_to_gray(img: np.ndarray):
    return np.dot(img[...], RGB_WEIGHTS)


class HyperParameters:
    def __init__(self, _dict):
        self._dict = _dict
        self._dict['DATA_DIR'] = str(pathlib.Path(__file__).parent.absolute() / self._dict['DATA_DIR'])

    def __getattr__(self, item):
        if item in self._dict:
            return self._dict[item]
        return self.__getattribute__(item)

    def __getitem__(self, item):
        return self._dict[item]


def load_hyperparameters(filename) -> HyperParameters:
    with open(filename) as f:
        return HyperParameters(yaml.load(f, Loader=LOADER))


def get_loaders(datadir: str, batch_size: int, num_workers=4, pin_memory=True):
    with h5py.File(f'{datadir}/train.h5') as h5:
        if 'cell' in datadir:
            images = np.stack([rgb_to_gray(h5['images'][i, ...].T).T for i in range(h5['images'].shape[0])])
            images = images.reshape((images.shape[0], 1, *images.shape[1:]))
        else:
            images = h5['images'][:]
        train_ds = DatasetLoader(images.astype(np.float32), h5['labels'][...].astype(np.float32))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    with h5py.File(f'{datadir}/valid.h5') as h5:
        if 'cell' in datadir:
            images = np.stack([rgb_to_gray(h5['images'][i, ...].T).T for i in range(h5['images'].shape[0])])
            images = images.reshape((images.shape[0], 1, *images.shape[1:]))
        else:
            images = h5['images'][:]
        val_ds = DatasetLoader(images.astype(np.float32), h5['labels'][...].astype(np.float32))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    torch.save(state, filename)


def save_predictions_as_imgs(epochidx, loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    # get the first batch only:
    input_images, true_denisty_map = next(iter(loader))
    input_images = input_images.to(device=device)
    true_denisty_maps = true_denisty_map.to(device)
    with torch.no_grad():
        preds = model(input_images)

    predicted_density_map = preds.cpu().detach().numpy().squeeze()
    true_denisty_maps = true_denisty_maps.cpu().detach().numpy().squeeze()

    fig, axs = plt.subplots(3, input_images.shape[0], sharex=True, sharey=True)
    for i in range(input_images.shape[0]):
        axs[0][i].imshow(input_images[i, 0, ...], cmap='gray', vmin=0, vmax=1)
        axs[1][i].imshow(true_denisty_maps[i, ...], cmap='gray', vmin=0, vmax=1)
        axs[2][i].imshow(predicted_density_map[i, ...], cmap='gray', vmin=0, vmax=1)
    axs[0][0].set_aspect('equal')
    plt.draw()
    plt.savefig(pathlib.Path(folder) / f'pred_{epochidx}.png')
    plt.close()

    model.train()


def evaluate_accuracy(loader, model, device) -> Dict:
    """Compute errors"""
    model.eval()
    true_counts = []
    predicted_counts = []

    # get prediction:
    with torch.no_grad():
        for _image, _density_map in loader:
            image = _image.to(device)
            density_map = _density_map.to(device)
            predicted_density_map = model(image)
            for true, predicted in zip(density_map, predicted_density_map):
                true_counts.append(torch.sum(true).item() / 100)
                predicted_counts.append(torch.sum(predicted).item() / 100)
    size = np.multiply(*tuple(image.shape[-2:]))
    err = [true - predicted for true, predicted in zip(true_counts, predicted_counts)]
    abs_err = [abs(error) for error in err]
    mean_err = sum(err) / size
    mean_abs_err = sum(abs_err) / size
    std = np.array(err).std()
    return dict(true_counts=true_counts, predicted_counts=predicted_counts,
                err=err, abs_err=abs_err, mean_err=mean_err, mean_abs_err=mean_abs_err, std=std)