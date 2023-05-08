import torch

from geometry import index, orthogonal, perspective
import torch.nn as nn
import torch.nn.functional as F
from filter import HourGlassFilter
from mlp import MLP
from network_util import init_network


class BasePIFuNetwork(nn.Module):
    def __init__(self, projection_mode='orthogonal', criteria={'occ': nn.MSELoss}):
        super().__init__()

        self.name = 'base_network'
        self.criteria = criteria

        self.projection = None
        self.index = None

        self.predictions = None
        self.predictions_intermediate = None
        self.predictions_down = None
        self.labels = None
        self.normals = None
        self.labels_normal = None
        self.preds_surface = None

    def filter(self, images):
        return

    def query(self, points, calibrations, transforms=None, labels=None):
        return

    def get_prediction(self):
        return self.preds



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
        self.mlp = MLP(filter_channels=self.opt.mlp_dim,
                       merge_layer=-1,
                       res_layers=self.opt.mlp_res_layers,
                       use_group_norm=self.opt.mlp_group_norm,
                       last_op=nn.Sigmoid())

        self.image_feature_list = []
        self.prediction_intermediate = None
        self.prediction_down = None
        self.w = None
        self.gamma = None

        self.intermediate_prediction_list = []

        init_network(self)

        self.netG = netG

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.opt.train_full_pifu:
            self.netG.eval()
        return self

    def filter_global(self, images):
        '''
        apply a fully convolutional network to images.
        :param images: [B1, C, H, W]
        '''
        if self.opt.train_full_pifu:
            self.netG.filter(images)
        else:
            with torch.no_grad():
                self.netG.filter(images)

    def filter_local(self, images, rect=None):
        '''
        apply a fully convolutional network to images.
        :param images: [B1, B2, C, H, W]
        '''
        normals = []
        if self.netG.opt.use_front_normal:
            normals.append(self.netG.normalizeF)
        if self.netG.opt.use_back_normal:
            normals.append(self.netG.normalizeB)

        if len(normals):
            normals = torch.cat(normals, 1)
            normals = nn.Upsample(size=(self.opt.loadSizeBig, self.opt.loadSizeBig), mode='bilinear',
                                     align_corners=True)

            if rect is None:
                normals = normals[:, None].expand(-1, images.size(1), -1, -1, -1)
                images = torch.cat([images, normals], 2)
            else:
                normal = []
                for i in range(rect.size(0)):
                    for j in range(rect.size(1)):
                        x1, x2, y1, y2 = rect[i, j]
                        tmp = normals[i, :, y1:y2, x1:x2]
                        normal.append(normals[i, :, y1:y2, x1:x2])
                normal = torch.stack(normal, 0).view(*rect.shape[:2], *normal[0].size())
                images = torch.cat([images, normal], 2)

        self.image_feature_list, self.normx = self.image_filter(images.view(-1, *images.size()[2:]))
        if not self.training:
            self.image_feature_list = [self.image_feature_list[:-1]]

    def query(self, points, calibration_local, calibration_global=None, transforms=None, labels=None):
        '''
        given 3d points, we obtain 2d projection of these given camera matrices.
        filter needs to be called beforehand.
        :param points: [B1, B2, 3, N] 3d points in world space
        :param calibration_local: [B1, B2, 4, 4] calibration matrices for each image
        :param calibration_global: [B1, 4, 4] calibration matrices for each image
        :param transforms: [B1, 2, 3] image space coordinate transforms
        :param labels: [B1, B2, C, W] ground truth labels
        :return: [B, C, N] prediction
        '''

        if calibration_global is not None:
            B = calibration_local.size(1)
        else:
            B = 1
            points = points[:, None]
            calibration_global = calibration_local
            calibration_local = calibration_local[:, None]

        ws = []
        predictions = []
        predictions_intermediate = []
        predictions_down = []
        gammas = []
        new_labels = []
        for i in range(B):
            xyz = self.projection(points[:, i], calibration_local[:, i], transforms)
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            in_bounding_box = (xyz >= -1) & (xyz <= 1)
            in_bounding_box = in_bounding_box[:, 0, :] & in_bounding_box[:, 1, :]
            in_bounding_box = in_bounding_box[:, None, :].detach().float()

            self.netG.query(points=points[:, i], calibration=calibration_global)
            predictions_down.append(torch.stack(self.netG.intermediate_prediction_list, 0))

            if labels is not None:
                new_labels.append(in_bounding_box * labels[:, i])
                with torch.no_grad():
                    ws.append(in_bounding_box.size(2) / in_bounding_box.view(in_bounding_box.size(0), -1).sum(1))
                    gammas.append(1 - new_labels[-1].view(new_labels.size(0), -1).sum(1)
                                  / in_bounding_box.view(in_bounding_box.size(0), -1).sum(1))

            z_feature = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feature = z_feature.detach()

            intermediate_prediction_list = []
            for j, image_feature in enumerate(self.image_feature_list):
                point_local_feature_list = [self.index(image_feature.view(-1, B, *image_feature.size()[1:]), xy),
                                            z_feature]
                point_local_feature = torch.cat(point_local_feature_list, 1)
                prediction = self.mlp(point_local_feature)[0]
                prediction = in_bounding_box * prediction
                intermediate_prediction_list.append(prediction)

            predictions_intermediate.append(torch.stack(intermediate_prediction_list, 0))
            predictions.append(intermediate_prediction_list[-1])

        self.predictions = torch.cat(predictions, 0)
        self.predictions_intermediate = torch.cat(predictions_intermediate, 1)
        self.predictions_down = torch.cat(predictions_down, 1)

        if labels is not None:
            self.w = torch.cat(ws, 0)
            self.gamma = torch.cat(gammas, 0)
            self.labels = torch.cat(new_labels, 0)

    def calculate_normal(self,
                         points,
                         calibration_local,
                         calibration_global,
                         transforms=None,
                         labels=None,
                         delta=.001,
                         fd_type='forward'):
        '''
        Return surface normal in 'model' space.
        :param points: [B1, B2, 3, N] 3d points in world space
        :param calibration_local: [B1, B2, 4, 4] calibration matrices for each image
        :param calibration_global: [B1, 4, 4] calibration matrices for each image
        :param transforms: [B1, 2, 3] image space coordinate transforms
        :param labels: [B1, B2, 3, N] ground truth normal
        :param delta: perturbation for finite difference
        :param fd_type: finite difference type (forward / backward / central)
        :return:
        '''
        B = calibration_local.size(1)

        if labels is not None:
            self.labels_normal = labels.view(-1, *labels.size()[2:])

        image_feature = self.image_feature_list[-1].view(-1, B, *self.image_feature_list[-1].size()[1:])

        normals = []
        for i in range(B):
            points_sub = points[:, i]
            pdx = points_sub.clone()
            pdx[:, 0, :] += delta
            pdy = points_sub.clone()
            pdy[:, 1, :] += delta
            pdz = points_sub.clone()
            pdx[:, 2, :] += delta

            points_all = torch.stack([points_sub, pdx, pdy, pdz], 3)
            points_all = points_all.view(*points_sub.size()[:2], -1)
            xyz = self.projection(points_all, calibration_local[:, i], transforms)
            xy = xyz[:, :2, :]

            self.netG.query(points=points_all, calibration=calibration_global, update_prediction=False)
            z_feature = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feature = z_feature.detach()

            point_local_feature_list = [self.index(image_feature[:, i], xy), z_feature]
            point_local_feature = torch.cat(point_local_feature_list, 1)
            prediction = self.mlp(point_local_feature)[0]

            prediction = prediction.view(*prediction.size()[:2], -1, 4)  # (B, 1, N, 4)

            # Divide by delta is omitted since it's normalized anyway
            dfdx = prediction[:, :, :, 1] - prediction[:, :, :, 0]
            dfdy = prediction[:, :, :, 2] - prediction[:, :, :, 0]
            dfdz = prediction[:, :, :, 3] - prediction[:, :, :, 0]

            normal = -torch.cat([dfdx, dfdy, dfdz], 1)
            normal = F.normalize(normal, dim=1, eps=1e-8)

            normals.append(normal)

        self.normals = torch.stack(normals, 1).view(-1, 3, points.size(3))

    def get_image_feature(self):
        '''
        Return the image filter in the last stack
        :return: [B, C, H, W]
        '''
        return self.image_feature_list[-1]

    def get_error(self):
        '''
        Return the loss given the ground truth labels and prediction
        '''

        error = {}

        if self.opt.train_full_pifu:
            if not self.opt.no_intermediate_loss:
                error['Err(occ)'] = 0.
                for i in range(self.predictions_down.size(0)):
                    error['Err(occ)'] += self.criteria['occ'](self.predictions_down[i],
                                                              self.labels,
                                                              self.gamma,
                                                              self.w)
                error['Err(occ)'] /= self.predictions_down.size(0)

            error['Err(occ:fine)'] = 0.
            for i in range(self.predictions_intermediate.size(0)):
                error['Err(occ:fine)'] += self.criteria['occ'](self.predictions_intermediate[i],
                                                               self.labels,
                                                               self.gamma,
                                                               self.w)
            error['Err(occ:fine)'] /= self.predictions_intermediate.size(0)

            if self.normlizes is not None and self.labels_normal is not None:
                error['Err(nml:fine)'] = self.criteria['normal'](self.normals, self.labels_normal)
        else:
            error['Err(occ:fine)'] = 0.
            for i in range(self.predictions_intermediate.size(0)):
                error['Err(occ:fine)'] += self.criteria['occ'](self.predictions_intermediate[i],
                                                               self.labels,
                                                               self.gamma,
                                                               self.w)
            error['Err(occ:fine)'] /= self.predictions_intermediate.size(0)

            if self.normals is not None and self.labels_normal is not None:
                error['Err(nml:fine)'] = self.criteria['normal'](self.normals, self.labels_normal)

        return error

    def forward(self,
                images_local,
                images_global,
                points,
                calibration_local,
                calibration_global,
                labels,
                points_normal=None,
                labels_normal=None,
                rect=None
                ):
        self.filter_global(images_global)
        self.filter_local(images_local)
        self.query(points, calibration_local, calibration_global, labels=labels)
        if points_normal is not None and labels_normal is not None:
            self.calculate_normal(points_normal, calibration_local, calibration_global, labels=labels_normal)
        result = self.get_prediction()
        error = self.get_error()

        return error, result
