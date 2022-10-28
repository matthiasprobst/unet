import pathlib
import re
from typing import Dict, Tuple, Callable, Union, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import yaml
from torch.utils.data import DataLoader

from ._logger import logger
from .dataset import DatasetLoader

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


def load_hyperparameters(yaml_filename) -> omegaconf.DictConfig:
    """Read a yaml file with package omegaconf and return as type DictConfg"""
    with open(yaml_filename, 'r') as f:
        return omegaconf.OmegaConf.load(f)


def get_loaders(train_filepath: pathlib.Path,
                valid_filepath: pathlib.Path,
                batch_size: int, num_workers: int = 4,
                pin_memory: bool = True,
                make_gray_scale: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Load data from HDF5"""
    train_filepath = pathlib.Path(train_filepath)
    valid_filepath = pathlib.Path(valid_filepath)

    with h5py.File(train_filepath) as h5:
        _images = h5['images'][:]
        if _images.shape[1] > 1 and make_gray_scale:
            images = np.stack([rgb_to_gray(_images[i, ...].T).T for i in range(_images.shape[0])])
            images = images.reshape((images.shape[0], 1, *images.shape[1:]))
        else:
            images = _images
        train_ds = DatasetLoader(images.astype(np.float32),
                                 h5['labels'][:].astype(np.float32))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    with h5py.File(valid_filepath) as h5:
        _images = h5['images'][:]
        if _images.shape[1] > 1 and make_gray_scale:
            images = np.stack([rgb_to_gray(_images[i, ...].T).T for i in range(_images.shape[0])])
            images = images.reshape((images.shape[0], 1, *images.shape[1:]))
        else:
            images = _images
        val_ds = DatasetLoader(images.astype(np.float32),
                               h5['labels'][...].astype(np.float32))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    """Save the checkpoint"""
    logger.debug('Saving checkpoint at %s', filename)
    torch.save(state, filename)


def _default_prediction_plot(images: np.ndarray,
                             labels: np.ndarray,
                             predictions: np.ndarray) -> List[plt.Figure]:
    """The default way of plotting the prediction

    Parameters
    ----------
    images: np.ndarray[n_images, m, ny, nx]
        Input images
    labels: np.ndarray[n_images, 1, ny, nx]
        Label
    predictions: np.ndarray[n_images, m, ny, nx]
        Label

    Returns
    -------
    figs: List[plt.Figure]
        List of figures
    """
    figs = []
    for i in range(images.shape[0]):
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        axs[0].imshow(images[i, 0, ...], cmap='gray', vmin=0, vmax=1)
        axs[1].imshow(labels[i, ...], cmap='gray', vmin=0, vmax=1)
        axs[2].imshow(predictions[i, ...], cmap='gray', vmin=0, vmax=1)
        axs[0].set_aspect('equal')
        plt.draw()
        figs.append(fig)
    return figs


def save_predictions_as_imgs(epochidx: int,
                             loader,
                             model,
                             fn: Callable = _default_prediction_plot,
                             folder: Union[pathlib.Path, str] = "saved_images/",
                             device: str = "cuda",
                             **fn_kwargs):
    """Save prediction(s) as image(s)"""
    # set model to evaluation mode
    model.eval()

    # get the first batch only:
    input_images, true_denisty_map = next(iter(loader))
    input_images = input_images.to(device=device)
    true_denisty_maps = true_denisty_map.to(device)
    with torch.no_grad():
        preds = model(input_images)

    predicted_density_map = preds.cpu().detach().numpy().squeeze()
    true_denisty_maps = true_denisty_maps.cpu().detach().numpy().squeeze()

    fig = fn(input_images, true_denisty_maps, predicted_density_map, **fn_kwargs)

    if isinstance(fig, (tuple, list)):
        fig_dir = pathlib.Path(folder) / f'pred_{epochidx:06d}'
        fig_dir.mkdir(exist_ok=True, parents=True)
        _fmt = f'0{len(str(len(fig)))}d'
        for ifig, f in enumerate(fig):
            f.savefig(fig_dir / f'pred_{ifig:{_fmt}}')
            plt.close()
    else:
        img_path = pathlib.Path(folder) / f'pred_{epochidx:06d}.png'
        logger.debug('saving prediciton images at %s', img_path)

        fig.savefig(img_path)
        plt.close()

    # set model to training mode
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
