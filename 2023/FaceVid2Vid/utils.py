import torch.jit
import yaml


def load_config(experiment_name, train_dir='train'):
    with open(f'{train_dir}/{experiment_name}.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


def normalize_vgg_face_tensor(tensor):
    mean = tensor.new_tensor([129.186279296875, 104.76238250732422, 93.59396362304688]).view(1, 3, 1, 1)
    std = tensor.new_tensor([1, 1, 1]).view(1, 3, 1, 1)
    normalized = (tensor * 255 - mean) / std
    return normalized


def normalize_vgg19_tensor(tensor):
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    return normalized


@torch.jit.script
def mean_min_value(x, is_positive=True):
    if is_positive:
        min_value = torch.min(x - 1, 0)
        loss = -torch.mean(min_value)
        return loss
    else:
        min_value = torch.min(-x - 1, 0)
        loss = -torch.mean(min_value)
        return loss
