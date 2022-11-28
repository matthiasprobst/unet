import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import torch.cuda.amp
import warnings
from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Union, Dict, Tuple

from ._logger import logger
from .model import UNET
from .utils import (get_loaders,
                    save_checkpoint,
                    save_predictions_as_imgs,
                    load_checkpoint,
                    PredictionPlot,
                    load_hyperparameters)

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')

file_dir = pathlib.Path(__file__).parent

hydra.verbose = True


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device) -> float:
    """Run training on one epoch"""
    loop = tqdm(loader)
    model.train()
    running_loss = 0.
    for data, targets in loop:
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # foward
        # make prediction:
        predictions = model(data)
        # compute loss
        loss = loss_fn(predictions, targets)
        running_loss += loss.item()
        # backward
        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update tqdm loop
        loop.set_postfix(train_loss=loss.item())

    return running_loss / len(loader)


def validate_one_epoch(loader, model, loss_fn, device) -> Tuple[float, Dict]:
    """Runs validation and compute errors"""
    loop = tqdm(loader)
    model.eval()
    true_counts = []
    predicted_counts = []

    running_loss = 0.
    with torch.no_grad():
        for _image, _density_map in loader:
            image = _image.to(device)
            density_map = _density_map.to(device)
            # get prediction:
            predicted_density_map = model(image)
            validation_loss = loss_fn(predicted_density_map, density_map)
            running_loss += validation_loss

            for true, predicted in zip(density_map, predicted_density_map):
                true_counts.append(torch.sum(true).item() / 100)
                predicted_counts.append(torch.sum(predicted).item() / 100)

            # update tqdm loop
            loop.set_postfix(valid_loss=validation_loss.item())

    size = np.multiply(*tuple(image.shape[-2:]))
    err = [true - predicted for true, predicted in zip(true_counts, predicted_counts)]
    abs_err = [abs(error) for error in err]
    mean_err = sum(err) / size
    mean_abs_err = sum(abs_err) / size
    std = np.array(err).std()
    return running_loss / len(loader), dict(true_counts=true_counts,
                                            predicted_counts=predicted_counts,
                                            err=err,
                                            abs_err=abs_err,
                                            mean_err=mean_err,
                                            mean_abs_err=mean_abs_err,
                                            std=std)


@dataclass
class Case:
    """Case class to control training and validation (tbd)"""
    cfg: Union[str, pathlib.Path, DictConfig]
    working_dir: Union[pathlib.Path, None] = None
    loss_dir: Union[pathlib.Path, None] = None
    checkpoints_dir: Union[pathlib.Path, None] = None
    plots_dir: Union[pathlib.Path, None] = None
    predicted_labels_dir: Union[pathlib.Path, None] = None
    current_best_mean_abs_err: float = np.infty
    current_epoch: int = 0
    prediction_plot_class: PredictionPlot = PredictionPlot

    def __post_init__(self):
        """post init call"""
        if not isinstance(self.cfg, DictConfig):
            self.cfg = load_hyperparameters(self.cfg)
        if self.working_dir is None:
            self.working_dir = pathlib.Path().cwd()
        else:
            self.working_dir = pathlib.Path(self.working_dir)

        if self.loss_dir is None:
            self.loss_dir = self.working_dir / 'loss'
        else:
            self.loss_dir = pathlib.Path(self.working_dir)

        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.working_dir / 'checkpoints'
        else:
            self.checkpoints_dir = pathlib.Path(self.working_dir)

        if self.predicted_labels_dir is None:
            self.predicted_labels_dir = self.working_dir / 'predicted_labels'
        else:
            self.predicted_labels_dir = pathlib.Path(self.working_dir)

        if self.plots_dir is None:
            self.plots_dir = self.working_dir / 'plots'
        else:
            self.plots_dir = pathlib.Path(self.working_dir)

        self.cfg['train_filename'] = self.working_dir / self.cfg['train_filename']
        self.cfg['valid_filename'] = self.working_dir / self.cfg['valid_filename']

    @property
    def is_empty_case(self) -> bool:
        """if directory paths don't exist, case most likely is empty"""
        for _dir in (self.working_dir, self.loss_dir, self.checkpoints_dir, self.predicted_labels_dir):
            if not _dir.exists():
                return True

    @property
    def paths(self) -> Dict:
        """Return paths dictionary of case"""
        return dict(working_dir=self.working_dir,
                    loss_dir=self.loss_dir,
                    checkpoints_dir=self.checkpoints_dir,
                    predicted_labels_dir=self.predicted_labels_dir,
                    plots_dir=self.plots_dir,
                    )

    def set_up_case(self):
        """creating folder structure"""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def reset_paths(self):
        """deleting case folder structure"""
        for pname, path in self.paths.items():
            if pname != 'working_dir' and path.exists():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)

    @property
    def get_latest_checkpoint(self) -> pathlib.Path:
        """return latest checkpoint"""
        return sorted(self.checkpoints_dir.glob('*.pth.tar'))[-1]

    def run(self, checkpoint: Union[pathlib.Path, Dict, None] = None) -> None:
        """Start the training"""
        self.set_up_case()

        # init model:
        if self.cfg.device == 'cuda' and not torch.cuda.is_available():
            logger.info('CUDA is not available but set in the config. Switching device to CPU.')
            self.cfg.device = 'cpu'

        # get train and valid data:
        train_filename = pathlib.Path(self.cfg.train_filename).resolve()
        valid_filename = pathlib.Path(self.cfg.valid_filename).resolve()

        for _name, _path in zip(('training', 'validation'),
                                (train_filename, valid_filename)):
            logger.debug('Loading %s data from %s', _name, _path)
            if _path:
                if not _path.exists():
                    raise FileNotFoundError(f'Cannot find training file: {_path}')

        train_loader, val_loader = get_loaders(
            train_filename,
            valid_filename,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            batch_size=self.cfg.batch_size
        )

        _img0, _label0 = train_loader.dataset[0]
        if 'in_channels' not in self.cfg or self.cfg.in_channels is None:
            self.cfg.in_channels = _img0.shape[0]
            logger.info(f'Set in_channels to {self.cfg.in_channels}')
        if 'out_channels' not in self.cfg or self.cfg.out_channels is None:
            self.cfg.out_channels = _label0.shape[0]
            logger.info(f'Set out_channels to {self.cfg.out_channels}')
        if not _img0.shape[0] == self.cfg.in_channels:
            raise ValueError(f'Input channel is set to 1 but image has different size: {_img0.shape[0]}')
        if not _label0.shape[0] == self.cfg.out_channels:
            raise ValueError(f'Output channel is set to 1 but label has different size: {_label0.shape[0]}')

        model = UNET(in_channels=self.cfg.in_channels,
                     out_channels=self.cfg.out_channels,
                     features=self.cfg.features,
                     up_stride=self.cfg.up.stride,
                     up_kernel_size=self.cfg.up.kernel_size,
                     down_stride=self.cfg.down.stride,
                     down_kernel_size=self.cfg.down.kernel_size,
                     use_upsample=self.cfg.use_upsample).to(self.cfg.device)
        if self.cfg.loss_fn.lower() == 'mse':
            loss_fn = nn.MSELoss()  # loss function
        elif self.cfg.loss_fn.lower() == 'bcewithlogits':
            loss_fn = nn.BCEWithLogitsLoss()  # loss function
        else:
            raise ValueError(f'Unknown loss function {self.cfg.loss_fn}')
        if self.cfg.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.cfg.optimizer.learning_rate)
        elif self.cfg.optimizer.name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.cfg.optimizer.learning_rate,
                                        momentum=self.cfg.optimizer.opts.SGD.momentum,
                                        weight_decay=self.cfg.optimizer.opts.SGD.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.cfg.optimizer}')

        scaler = torch.cuda.amp.GradScaler()

        with open(self.loss_dir / 'loss.txt', 'w') as f:
            f.write('epoch,loss,training_loss,validation_loss')

        # a stp-file in the cwd will stop the epoch loop
        stp_file = pathlib.Path.cwd() / 'stp'

        if checkpoint is not None:  # resume a run
            if isinstance(checkpoint, (str, pathlib.Path)):
                if not pathlib.Path(checkpoint).exists():
                    raise FileNotFoundError(f'Checkpoint not found: {pathlib.Path(checkpoint).absolute()}')
            logger.info('Resuming from checkpoint %s', checkpoint)
            load_checkpoint(checkpoint, model)

        with SummaryWriter(log_dir=self.working_dir) as writer:
            for epoch in range(self.cfg.num_epochs):

                logger.info('Entering epoch %i/%i', epoch, self.cfg.num_epochs)

                self.current_epoch = epoch

                if stp_file.exists():
                    logger.info('Found stop signal. Not running epoch %s', epoch)
                    break

                # train one epoch and get average loss per batch:
                average_loss = train_one_epoch(
                    train_loader,
                    model,
                    optimizer,
                    loss_fn,
                    scaler,
                    self.cfg.device
                )
                writer.add_scalar('Loss/train', average_loss, epoch)
                writer.add_scalar('RunningLoss/train', epoch)

                # check accuracy/run validation:
                validation_average_loss, err_dict = validate_one_epoch(val_loader,
                                                                       model,
                                                                       loss_fn,
                                                                       self.cfg.device)

                logger.info(f'Epoch %i/%i: Training-Loss: %f | Training-Loss: %f',
                            epoch, self.cfg.num_epochs, average_loss, validation_average_loss)
                with open(self.loss_dir / 'loss.txt', 'a') as f:
                    f.write(f'\n{epoch}, {average_loss}, {validation_average_loss}')
                writer.add_scalar('Accuracy/mean_abs_err', err_dict['mean_abs_err'], epoch)
                logger.info('MAE: %f', err_dict["mean_abs_err"])

                if self.current_best_mean_abs_err > err_dict['mean_abs_err']:
                    logger.info('Generating true-vs-pediction plot')
                    fig = plt.figure()
                    plt.title(f'Epoch {epoch}')
                    plt.plot(err_dict['true_counts'], err_dict['predicted_counts'], 'k+')
                    plt.plot([min(err_dict['true_counts']), max(err_dict['true_counts'])],
                             [min(err_dict['true_counts']), max(err_dict['true_counts'])], 'k--')
                    plt.xlabel('true counts [-]')
                    plt.ylabel('predicted counts [-]')
                    plt.draw()
                    _fmt = f'0{len(str(self.cfg.num_epochs))}d'
                    plt.savefig(self.plots_dir / f'true_vs_predicted_{epoch:{_fmt}}', dpi=150)
                    writer.add_figure(str(self.plots_dir / 'true_vs_predicted'), fig, global_step=epoch)

                    self.current_best_mean_abs_err = err_dict['mean_abs_err']
                    # save model
                    logger.info(f'New best mean abs err: {err_dict["mean_abs_err"]}')
                    if self.cfg.save_checkpoint:
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict()
                        }
                        checkpoint_file = self.checkpoints_dir / f'cp{epoch}.pth.tar'
                        save_checkpoint(checkpoint, filename=checkpoint_file)

                    # print some examples to a folder
                    logger.debug('saving prediction images')
                    save_predictions_as_imgs(
                        epoch,
                        val_loader,
                        model,
                        folder=self.predicted_labels_dir,
                        device=self.cfg.device,
                        predplot=self.prediction_plot_class
                    )
