"""example file using unet with hydra"""
import hydra
from omegaconf import DictConfig

import unet


@hydra.main(config_path='.', config_name='hyperparameters.yml')
def main(cfg: DictConfig) -> None:
    """Main function running the unet base on the configuration"""
    case = unet.Case(cfg, None)
    case.run()


if __name__ == '__main__':
    main()
