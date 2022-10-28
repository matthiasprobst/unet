"""example file using unet with hydra"""
from typing import List

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import unet
from unet import utils


class PredictionPlot(utils.PredictionPlot):
    """Subclass of standard prediction plot class"""

    def plot(self) -> List[plt.Figure]:
        """adding titles to figures"""
        figs = super().plot()
        for i, fig in enumerate(figs):
            fig.axes[0].set_title(f'real img # {i}')
            fig.axes[1].set_title(f'{self.labels[i, :, :].sum() / 100:3.1f}')
            fig.axes[2].set_title(f'{self.predictions[i, :, :].sum() / 100:3.1f}')
        return figs


run1_wd = None


@hydra.main(config_path='.', config_name='hyperparameters.yaml')
def run1(cfg: DictConfig) -> None:
    """initial run"""
    global run1_wd
    case = unet.Case(cfg)
    case.prediction_plot_class = PredictionPlot
    run1_wd = case.working_dir
    case.run()


@hydra.main(config_path='.', config_name='hyperparameters.yaml')
def run2(cfg: DictConfig) -> None:
    """follow-up run"""
    global run1_wd
    # now resume onthe same configuration file in this case
    case = unet.Case(cfg, None)
    case.prediction_plot_class = PredictionPlot

    cp_dir = run1_wd / 'checkpoints'

    case.run(checkpoint=sorted(cp_dir.glob('*.pth.tar'))[-1])


if __name__ == '__main__':
    run1()
    run2()
