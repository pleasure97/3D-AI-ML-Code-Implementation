import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.model_zoo import load_url
from torchvision.models import vgg16
from torchvision.models import vgg19
from utils import normalize_vgg_face_tensor, normalize_vgg19_tensor, mean_min_value


class PerceptualNetwork(nn.Module):
    def __init__(self, model, layer_name, layers):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}

        for i, layer in enumerate(self.model):
            x = layer(x)
            layer_name = self.layer_name.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x

        return output


def get_vgg19(layers):
    model = vgg19()
    state_dict = load_url("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
                          map_location=torch.device("cpu"), progress=True)
    model.load_state_dict(state_dict)
    model = model.features
    layer_name = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        17: "relu_3_4",
        20: "relu_4_1",
        22: "relu_4_2",
        24: "relu_4_3",
        26: "relu_4_4",
        29: "relu_5_1"
    }

    return PerceptualNetwork(model, layer_name, layers)


def get_vgg_face(layers):
    model = vgg16(num_classes=2622)
    state_dict = load_url("http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/" "vgg_face_dag.pth",
                          map_location=torch.device("cpu"), progress=True)
    feature_layer_name = {
        0: "conv1_1",
        2: "conv1_2",
        5: "conv2_1",
        7: "conv2_2",
        10: "conv3_1",
        12: "conv3_2",
        14: "conv3_3",
        17: "conv4_1",
        19: "conv4_2",
        21: "conv4_3",
        24: "conv5_1",
        26: "conv5_2",
        28: "conv5_3"
    }

    classifier_layer_name = {
        0: "fc6",
        3: "fc7",
        6: "fc8"
    }

    layer_name = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        18: "relu_4_1",
        20: "relu_4_2",
        22: "relu_4_3",
        25: "relu_5_1"
    }

    new_state_dict = {}

    for key, value in feature_layer_name.items():
        new_state_dict["features." + str(key) + ".weight"] = state_dict[value + ".weight"]
        new_state_dict["features." + str(key) + ".bias"] = state_dict[value + ".bias"]

    for key, value in classifier_layer_name.items():
        new_state_dict["classifier." + str(key) + ".weight"] = state_dict[value + ".weight"]
        new_state_dict["classifier." + str(key) + ".bias"] = state_dict[value + ".bias"]

    model.load_state_dict(new_state_dict)
    model = model.features

    return PerceptualNetwork(model, layer_name, layers)


class PerceptualLoss(nn.Module):
    def __init__(self, layers_weight, num_scale=3):
        super().__init__()
        self.vgg19 = get_vgg19(layers_weight.keys()).eval()
        self.vgg_face = get_vgg_face(layers_weight.keys()).eval()
        self.criterion = nn.L1Loss()
        self.layers_weight = layers_weight
        self.num_scale = num_scale

    def forward(self, input_tensor, target_tensor):

        loss = 0.
        loss += self.criterion(input_tensor, target_tensor)

        vgg_face_input_features = self.vgg_face(normalize_vgg_face_tensor(input_tensor))
        vgg_face_target_features = self.vgg_face(normalize_vgg_face_tensor(target_tensor))

        input_tensor = normalize_vgg19_tensor(input_tensor)
        target_tensor = normalize_vgg19_tensor(target_tensor)

        vgg19_input_features = self.vgg19(input_tensor)
        vgg19_target_features = self.vgg19(target_tensor)

        for layer, weight in self.layers_weight.items():
            loss += weight * self.criterion(vgg_face_input_features[layer],
                                            vgg_face_target_features[layer].detach()) / 255.
            loss += weight * self.criterion(vgg19_input_features[layer], vgg19_target_features[layer].detach())

        for i in range(self.num_scale):
            input_tensor = F.interpolate(input_tensor, mode="bilinear",
                                         scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            target_tensor = F.interpolate(target_tensor, mode="bilinear",
                                          scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            vgg19_input_features = self.vgg19(input_tensor)
            vgg19_target_features = self.vgg19(target_tensor)
            loss += weight * self.criterion(vgg19_input_features[layer], vgg19_target_features[layer].detach())

        return loss


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator, use_real_label, update_discriminator=True):

        if update_discriminator:
            if use_real_label:
                loss = mean_min_value(discriminator, is_positive=True)
            else:
                loss = mean_min_value(discriminator, is_positive=False)
        else:
            loss = -torch.mean(discriminator)

        return loss


class EquivarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, detected_keypoints, transformed_keypoints):
        loss = self.criterion(detected_keypoints[:, :, :2], transformed_keypoints)
        return loss


class KeypointPriorLoss(nn.Module):
    def __init__(self, d_t=.1, z_t=.33):
        super().__init__()
        self.D_t = d_t
        self.z_t = z_t

    def forward(self, detected_keypoints):
        distance_matrix = torch.cdist(detected_keypoints, detected_keypoints).square()
        loss = torch.max(0, self.D_t - distance_matrix).sum((1, 2)).mean() \
               + torch.abs(detected_keypoints[:, :, 2].mean(1) - self.z_t).mean() \
               - detected_keypoints.shape[1] * distance_matrix
        return loss


class HeadPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred_yaw, true_yaw, pred_pitch, true_pitch, pred_roll, true_roll):
        loss = (self.criterion(pred_yaw, true_yaw.detach())
                + self.criterion(pred_pitch, true_pitch.detach())
                + self.criterion(pred_roll, true_roll.detach()))
        loss = loss / np.pi * 180.
        return loss


class DeformationPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_dk):
        loss = delta_dk.abs().mean()

        return loss
