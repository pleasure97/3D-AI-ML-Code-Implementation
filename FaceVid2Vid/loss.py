import torch.nn as nn

class FinalLoss(nn.Module):

    def __init__(self):
        self.loss_p = loss_p
        self.weight_p = 10.
        self.weight_g = 1.
        self.weight_e = 20.
        self.weight_l = 10.
        self.weight_h = 20.
        self.weight_delta = 5.

    return

def get_perceptual_loss(driving_image, output_image, weight = 10.):


    return loss_p

def get_gan_loss(driving_image, output_image):

    return loss_g

def get_equivariance_loss(x_dk):

    return loss_e

def get_keypoint_prior_loss(x_dk):

    return loss_l

def get_head_pose_loss(predicted_rotation, ground_truth_rotation):

    return loss_h

def get_deformation_prior_loss(delta_dk):

    return loss_delta 

