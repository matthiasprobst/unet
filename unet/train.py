import logging
import pathlib
import warnings
from typing import Union

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda.amp
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import UNET
from .utils import get_loaders, save_checkpoint, save_predictions_as_imgs, evaluate_accuracy

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')

file_dir = pathlib.Path(__file__).parent
logger = logging.getLogger('test')
hydra.verbose = True


def train_fn(loader, model, optimizer, loss_fn, scaler, device) -> float:
    """train function"""
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        # foward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return loss.item()


class Case:
    def __init__(self, cfg: DictConfig, working_dir: Union[pathlib.Path, None]):
        if working_dir is None:
            _working_dir = pathlib.Path().cwd()
        else:
            _working_dir = pathlib.Path(working_dir)
        self.paths = {'working_dir': _working_dir,
                      'loss': _working_dir / 'loss',
                      'checkpoints': _working_dir / 'checkpoints',
                      'prediced_labels': _working_dir / 'prediced_labels'}
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        self._config = cfg

        self.model = UNET(in_channels=1,
                          out_channels=1,
                          features=self._config.features,
                          up_stride=self._config.up.stride,
                          up_kernel_size=self._config.up.kernel_size,
                          down_stride=self._config.down.stride,
                          down_kernel_size=self._config.down.kernel_size,
                          use_upsample=self._config.use_upsample).to(self._config.device)
        self.loss_fn = nn.MSELoss()  # loss function
        if self._config.optimizer.name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self._config.optimizer.learning_rate)
        elif self._config.optimizer.name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self._config.optimizer.learning_rate,
                                             momentum=self._config.optimizer.opts.SGD.momentum,
                                             weight_decay=self._config.optimizer.opts.SGD.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self._config.optimizer}')

        train_filename = pathlib.Path(self._config.train_filename).resolve()
        if not train_filename.exists():
            raise FileNotFoundError(f'Cannot find training file: {train_filename}')
        valid_filename = pathlib.Path(self._config.valid_filename).resolve()
        if not valid_filename.exists():
            raise FileNotFoundError(f'Cannot find validation file: {valid_filename}')
        self.train_loader, self.val_loader = get_loaders(
            train_filename,
            valid_filename,
            batch_size=self._config.BATCH_SIZE
        )

        torch.cpu.amp.autocast_mode
        self.scaler = torch.cuda.amp.GradScaler()

        run_txt_filename = 'loss/loss.txt'
        with open(self.working_dir / run_txt_filename, 'w') as f:
            f.write('epoch,loss')

        self.current_best_mean_abs_err = np.infty

    @property
    def working_dir(self) -> pathlib.Path:
        """Return the working dir"""
        return self.paths['working_dir']

    def run(self) -> None:
        """Start the training"""
        stp_file = pathlib.Path.cwd() / 'stp'
        with SummaryWriter() as writer:
            for epoch in range(self._config.NUM_EPOCHS):
                if stp_file.exists():
                    logger.info('Found stop signal. Not running epoch %s', epoch)
                    break
                logger.info('Epoch %s', epoch)
                loss = train_fn(
                    self.train_loader,
                    self.model,
                    self.optimizer,
                    self.loss_fn,
                    self.scaler,
                    self._config.device
                )
                writer.add_scalar('Loss/train', loss, epoch)
                with open(self.paths['loss'] / 'loss.txt', 'a') as f:
                    f.write(f'\n{epoch}, {loss}')
                logger.info('Loss: %f', loss)

                # check accuracy:
                err_dict = evaluate_accuracy(self.val_loader, self.model, self._config.device)
                writer.add_scalar('Accuracy/mean_abs_err', err_dict['mean_abs_err'], epoch)
                logger.info(f'MAE: {err_dict["mean_abs_err"],}')

                if self.current_best_mean_abs_err > err_dict['mean_abs_err']:
                    logger.info('Generating true-vs-pediction plot')
                    fig = plt.figure()
                    plt.title(f'epoch {epoch}')
                    plt.plot(err_dict['true_counts'], err_dict['predicted_counts'], 'k+')
                    plt.plot([min(err_dict['true_counts']), max(err_dict['true_counts'])],
                             [min(err_dict['true_counts']), max(err_dict['true_counts'])], 'k--')
                    plt.xlabel('true counts [-]')
                    plt.ylabel('predicted counts [-]')
                    writer.add_figure('plots/true_vs_predicted', fig, global_step=epoch)

                    self.current_best_mean_abs_err = err_dict['mean_abs_err']
                    # save model
                    logger.info(f'New best mean abs err: {err_dict["mean_abs_err"]}')
                    if self._config.save_checkpoint:
                        checkpoint = {
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict()
                        }
                        checkpoint_file = self.paths['checkpoints'] / f'cp{epoch}.pth.tar'
                        logger.info('Saving checkoint: %s', checkpoint_file)
                        save_checkpoint(checkpoint, filename=checkpoint_file)

                    # print some examples to a folder
                    logger.debug('saving prediction images')
                    save_predictions_as_imgs(
                        epoch,
                        self.val_loader,
                        self.model,
                        folder=self.paths['prediced_labels'],
                        device=self._config.device
                    )


@hydra.main(config_path='../tests/conf', config_name='hyperparameters.yml')
def main(cfg: DictConfig) -> None:
    """Main function running the unet base on the configuration"""
    case = Case(cfg, None)
    case.run()


if __name__ == '__main__':
    main()
