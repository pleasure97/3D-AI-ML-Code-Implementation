import torch
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(config_dict: DictConfig):
    pass