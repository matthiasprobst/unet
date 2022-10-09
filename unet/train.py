import logging
import pathlib
import warnings

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda.amp
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import UNET
from utils import get_loaders, save_checkpoint, save_predictions_as_imgs, evaluate_accuracy

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')

file_dir = pathlib.Path(__file__).parent
logger = logging.getLogger('test')
hydra.verbose = True


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
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


@hydra.main(config_path='conf', config_name='hyperparameters.yml')
def main(cfg: DictConfig) -> None:
    pathlib.Path('loss').mkdir(parents=True)
    pathlib.Path('checkpoints').mkdir(parents=True)
    pathlib.Path('prediced_labels').mkdir(parents=True)

    hp = cfg
    # hp = load_hyperparameters('conf/hyperparameters.yml')
    model = UNET(in_channels=1,
                 out_channels=1,
                 features=hp.features,
                 up_stride=hp.up.stride,
                 up_kernel_size=hp.up.kernel_size,
                 down_stride=hp.down.stride,
                 down_kernel_size=hp.down.kernel_size,
                 use_upsample=hp.use_upsample).to(hp.device)
    loss_fn = nn.MSELoss()  # loss function
    if hp.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.optimizer.learning_rate)
    elif hp.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hp.optimizer.learning_rate,
                                    momentum=hp.optimizer.opts.SGD.momentum,
                                    weight_decay=hp.optimizer.opts.SGD.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {hp.optimizer}')

    train_loader, val_loader = get_loaders(str(pathlib.Path(hydra.utils.get_original_cwd()) / hp.DATA_DIR),
                                           batch_size=hp.BATCH_SIZE)

    scaler = torch.cuda.amp.GradScaler()

    run_txt_filename = 'loss/loss.txt'
    with open(run_txt_filename, 'w') as f:
        f.write('epoch,loss')

    current_best_mean_abs_err = np.infty

    with SummaryWriter() as writer:
        for epoch in range(hp.NUM_EPOCHS):
            logger.info(f'Epoch {epoch}')
            loss = train_fn(
                train_loader,
                model,
                optimizer,
                loss_fn,
                scaler,
                hp.device
            )
            writer.add_scalar('Loss/train', loss, epoch)
            with open(run_txt_filename, 'a') as f:
                f.write(f'\n{epoch}, {loss}')
            logger.info(f'Loss: {loss}')

            # check accuracy:
            err_dict = evaluate_accuracy(val_loader, model, hp.device)
            writer.add_scalar('Accuracy/mean_abs_err', err_dict['mean_abs_err'], epoch)
            logger.info(f'MAE: {err_dict["mean_abs_err"],}')

            if current_best_mean_abs_err > err_dict['mean_abs_err']:
                logger.info('Generating true-vs-pediction plot')
                fig = plt.figure()
                plt.title(f'epoch {epoch}')
                plt.plot(err_dict['true_counts'], err_dict['predicted_counts'], 'k+')
                plt.plot([min(err_dict['true_counts']), max(err_dict['true_counts'])],
                         [min(err_dict['true_counts']), max(err_dict['true_counts'])], 'k--')
                plt.xlabel('true counts [-]')
                plt.ylabel('predicted counts [-]')
                writer.add_figure('plots/true_vs_predicted', fig, global_step=epoch)

                current_best_mean_abs_err = err_dict['mean_abs_err']
                # save model
                logger.info(f'New best mean abs err: {err_dict["mean_abs_err"]}')
                if hp.save_checkpoint:
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    logger.info(f'Saving checkoint: checkpoints/cp{epoch}.pth.tar')
                    save_checkpoint(checkpoint, filename=f'checkpoints/cp{epoch}.pth.tar')

                # print some examples to a folder
                logger.debug('saving prediction images')
                save_predictions_as_imgs(
                    epoch, val_loader, model, folder="prediced_labels", device=hp.device
                )


if __name__ == '__main__':
    main()
