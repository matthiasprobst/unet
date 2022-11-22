import logging
import pathlib
import re
from typing import Dict, Tuple, Union, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import DatasetLoader

logger = logging.getLogger('unet')
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
                make_gray_scale: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load data from HDF5."""
    train_filepath = pathlib.Path(train_filepath)
    valid_filepath = pathlib.Path(valid_filepath)

    def _get_from_hdf(filepath):
        with h5py.File(filepath) as h5:
            _images = h5['images'][:]
            if _images.shape[1] > 1 and make_gray_scale:
                images = np.stack([rgb_to_gray(_images[i, ...].T).T for i in range(_images.shape[0])])
                images = images.reshape((images.shape[0], 1, *images.shape[1:]))
            else:
                images = _images
            return DatasetLoader(images.astype(np.float32), h5['labels'][:].astype(np.float32))

    train_ds = _get_from_hdf(train_filepath)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = _get_from_hdf(valid_filepath)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar") -> pathlib.Path:
    """Save the checkpoint"""
    logger.debug('Saving checkpoint at %s', filename)
    torch.save(state, filename)
    return pathlib.Path(filename)


def load_checkpoint(checkpoint: Union[pathlib.Path, Dict], model):
    """load from checkpoint file"""
    if isinstance(checkpoint, (str, pathlib.Path)):
        model.load_state_dict(torch.load(checkpoint)["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])


class PredictionPlot:

    def __init__(self, images: np.ndarray,
                 labels: np.ndarray,
                 predictions: np.ndarray, ):
        """
        Parameters
        ----------
        images: np.ndarray[n_images, m, ny, nx]
            Input images
        labels: np.ndarray[n_images, 1, ny, nx]
            Label
        predictions: np.ndarray[n_images, m, ny, nx]
            Label
        """
        self.images = images
        self.labels = labels
        self.predictions = predictions

    def plot(self) -> List[plt.Figure]:
        """The default way of plotting the prediction

        Returns
        -------
        figs: List[plt.Figure]
            List of figures
        """
        figs = []
        for i in range(self.images.shape[0]):
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            axs[0].imshow(self.images[i, 0, ...], cmap='gray', vmin=0, vmax=1)
            axs[1].imshow(self.labels[i, ...], cmap='gray', vmin=0, vmax=1)
            axs[2].imshow(self.predictions[i, ...], cmap='gray', vmin=0, vmax=1)
            axs[0].set_aspect('equal')
            plt.draw()
            figs.append(fig)
        return figs


default_prediction_plot_class = PredictionPlot


def save_predictions_as_imgs(epochidx: int,
                             loader,
                             model,
                             predplot: PredictionPlot = default_prediction_plot_class,
                             folder: Union[pathlib.Path, str] = "saved_images/",
                             device: str = "cuda"):
    """Save prediction(s) as image(s)"""
    # set model to evaluation mode
    model.eval()

    # get the first batch only:
    input_images, true_density_map = next(iter(loader))

    input_images = input_images.to(device=device)
    true_density_maps = true_density_map.to(device=device)
    with torch.no_grad():
        preds = model(input_images)

    predicted_density_map = preds.cpu().detach().numpy().squeeze()
    true_density_maps = true_density_maps.cpu().detach().numpy().squeeze()

    fig = predplot(input_images.cpu(), true_density_maps, predicted_density_map).plot()

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
