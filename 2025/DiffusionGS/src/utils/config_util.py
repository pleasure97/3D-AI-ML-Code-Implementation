from omegaconf import DictConfig

def set_config(new_config: DictConfig) -> None:
    global config
    config = new_config

def get_config() -> DictConfig:
    global config
    return config