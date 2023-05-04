from geometry import index, orthogonal, perspective
import torch.nn as nn
from filter import HourGlassFilter

class BasePIFuNetwork(nn.Module):
    def __init__(self, projection_mode='orthogonal', criteria={'occ': nn.MSELoss}):
        super().__init__()

        self.name = 'base_network'
        self.criteria = criteria

        self.projection = None
        self.index = None

        self.preds = None
        self.labels = None
        self.nmls = None
        self.labels_nml = None
        self.preds_surface = None

    def filter(self, images):
        '''
        Apply a fully convolutional network to images.
        :param images: [B, C, H, W]
        '''
        return

    def query(self, points, calibrations, transforms=None, labels=None):
        '''
        Given 3d points, we obtain 2d projection of these given camera matrices.
        :param points: [B, 3, N] 3D points in world space
        :param calibrations: [B, 3, 4] Calibration matrices for each image
        :param transforms: [B, 2, 3] Image space coordinate transforms
        :param labels: [B, C, N] Ground truth labels (for supervision only)
        :return: prediction : [B, C, N]
        '''
        return

    def get_prediction(self):
        '''
        Return the current prediction.
        :return: prediction : [B, C, N]
        '''
        return self.preds

    def forward(self, points, images, calibrations, transforms=None):
        self.filter(images)
        self.query(points, calibrations, transforms)
        return self.get_prediction()

    def calculate_normal(self, points, calibrations, transforms=None, delta=.1):
        '''
        Return surface normal in 'model' space.
        Computes normal only in the last stack.
        :param points: [B, 3, N] 3D Points in world space
        :param calibrations: [B, 3, 4] Calibration matrices for each image
        :param transforms: [B, 2, 3] Image space coordinate transforms
        :param delta: Perturbation for finite difference
        :return:
        '''
        return


class HourGlassPIFuNetwork(BasePIFuNetwork):
    def __init__(self,
                 opt,
                 netG,
                 block_options,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss},
                 ):
        super().__init__()
        self.name = 'HourGlassPIFu'

        in_channel = 3

        if netG.opt.use_front_normal_map:
            in_channel += 3
        if netG.opt.use_back_normal_map:
            in_channel += 3

        self.opt = opt
        self.block_options = block_options
        self.image_filter = HourGlassFilter(opt.num_stack,
                                            opt.depth,
                                            in_channel,
                                            opt.hg_dim,
                                            block_options,
                                            down_type='no_down',
                                            use_sigmoid=False)
        
