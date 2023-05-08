'''
MIT License
Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import cv2
import numpy as np
from geometry_util import orthogonal


class Camera:
    def __init__(self, width=1600., height=1020.):
        # Focal Length
        # equivalent to 50mm
        focal = np.sqrt(width * width + height * height)  # 2000
        self.focal_x = focal
        self.focal_y = focal
        # Principal Point Offset
        self.principal_x = width / 2
        self.principal_y = width / 2
        # Axis skew
        self.skew = 0.
        # Image size
        self.width = width
        self.height = height

        self.near = 1.
        self.far = 10.

        # Camera center
        self.eye = np.array([0., 0., -3.6])
        self.center = np.array([0., 0., 0.])
        self.direction = np.array([0., 0., -1])
        self.right = np.array([1., 0., 0.])
        self.up = np.array([0., 1., 0.])

        self.orthogonal_ratio = None

    def sanity_check(self):
        self.center = self.center.reshape([-1])
        self.direction = self.direction.reshape([-1])
        self.right = self.right.reshape([-1])
        self.up = self.up.reshape([-1])

        assert len(self.center) == 3
        assert len(self.direction) == 3
        assert len(self.right) == 3
        assert len(self.up) == 3

    @staticmethod
    def normalize_vector(vec):
        vec_norm = np.linalg.norm(vec)
        return vec if vec_norm == 0 else vec / vec_norm

    def get_real_z_value(self, z):
        z_near = self.near
        z_far = self.far
        z_n = 2. * z - 1.
        z_e = 2. * z_near * z_far / (z_far + z_near - z_n * (z_far - z_near))
        return z_e

    def get_rotation_matrix(self):
        rotation_matrix = np.identity(3)
        d = self.eye - self.center
        d = -self.normalize_vector(d)
        u = self.up
        self.right = -np.cross(u, d)
        u = np.cross(d, self.right)
        rotation_matrix[0, :] = self.right
        rotation_matrix[1, :] = u
        rotation_matrix[2, :] = d

        return rotation_matrix

    def get_translation_vector(self):
        rotation_matrix = self.get_rotation_matrix()
        translation_vector = -np.dot(rotation_matrix.T, self.eye)

        return translation_vector

    def get_intrinsic_matrix(self):
        intrinsic_matrix = np.identity(3)

        intrinsic_matrix[0, 0] = self.focal_x
        intrinsic_matrix[1, 1] = self.focal_y
        intrinsic_matrix[0, 1] = self.skew
        intrinsic_matrix[0, 2] = self.principal_x
        intrinsic_matrix[1, 2] = self.principal_y

        return intrinsic_matrix

    def get_projection_matrix(self):
        rotation_matrix = self.get_rotation_matrix()
        translation_vector = self.get_translation_vector()

        extrinsic_matrix = np.identity(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector

        return extrinsic_matrix[:3, :]

    def set_rotation_matrix(self, rotation_matrix):
        self.direction = rotation_matrix[2, :]
        self.up = -rotation_matrix[1, :]
        self.right = rotation_matrix[0, :]

    def set_intrinsic_matrix(self, intrinsic_matrix):
        self.focal_x = intrinsic_matrix[0, 0]
        self.focal_y = intrinsic_matrix[1, 1]
        self.skew = intrinsic_matrix[0, 1]
        self.principal_x = intrinsic_matrix[0, 2]
        self.principal_y = intrinsic_matrix[1, 2]

    def set_projection_matrix(self, projection_matrix):
        result = cv2.decomposeProjectionMatrix(projection_matrix)
        intrinsic_matrix, rotation_matrix, camera_center_homo = result[0], result[1], result[2]
        camera_center = camera_center_homo[0:3] / camera_center_homo[3]
        camera_center = camera_center.reshape([-1])
        intrinsic_matrix /= intrinsic_matrix[2][2]

        self.set_intrinsic_matrix(intrinsic_matrix)
        self.set_rotation_matrix(rotation_matrix)
        self.center = camera_center

        self.sanity_check()

    def get_gl_matrix(self):
        z_near = self.near
        z_far = self.far
        rotation_matrix = self.get_rotation_matrix()
        intrinsic_matrix = self.get_intrinsic_matrix()
        translation_vector = self.get_translation_vector()

        extrinsic_matrix = np.identity(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector
        axis_adjacent = np.identity(4)
        axis_adjacent[2, 2] = -1.
        axis_adjacent[1, 1] = -1.
        model_view = np.matmul(axis_adjacent, extrinsic_matrix)

        projective = np.zeros([4, 4])
        projective[:2, :2] = intrinsic_matrix[:2, :2]
        projective[:2, 2:3] = -intrinsic_matrix[:2, 2:3]
        projective[3, 2] = -1
        projective[2, 2] = (z_near + z_far)
        projective[2, 3] = (z_near * z_far)

        if self.orthogonal_ratio is None:
            ndc = orthogonal(left=0, right=self.width, bottom=0, top=self.height, z_near=z_near, z_far=z_far)
            perspective_matrix = np.matmul(ndc, projective)
        else:
            perspective_matrix = orthogonal(left=(-self.width * self.orthogonal_ratio / 2),
                                            right=(self.width * self.orthogonal_ratio / 2),
                                            bottom=(-self.height * self.orthogonal_ratio / 2),
                                            top=(self.height * self.orthogonal_ratio / 2),
                                            z_near=z_near,
                                            z_far=z_far)

        return perspective_matrix, model_view


def KRT_from_P(projection_matrix, normalize_K=True):
    result = cv2.decomposeProjectionMatrix(projection_matrix)
    K, Rot, camera_center_homog = result[0], result[1], result[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    if normalize_K:
        K /= K[2][2]
    return K, Rot, trans


def MVP_from_P(projection_matrix, width, height, near=.1, far=10000.):
    '''
    Convert OpenCV camera calibration matrix to OpenGL projection and model view matrix
    :param projection_matrix: OpenCV camera projection matrix
    :param width: Image width
    :param height: Image height
    :param near: Z near value
    :param far: Z far value
    :return: OpenGL projection matrix and model view matrix
    '''

    result = cv2.decomposeProjectionMatrix(projection_matrix)
    K, Rot, camera_center_homog = result[0], result[1], result[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    K /= K[2][2]

    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3, :3] = Rot
    extrinsic_matrix[:3, 3:4] = trans
    axis_adjacent = np.identity(4)
    axis_adjacent[2, 2] = -1
    axis_adjacent[1, 1] = -1
    model_view = np.matmul(axis_adjacent, extrinsic_matrix)

    z_far = far
    z_near = near
    projective = np.zeros([4, 4])
    projective[:2, :2] = K[:2, :2]
    projective[:2, 2:3] = -K[:2, 2:3]
    projective[3, 2] = -1
    projective[2, 2] = (z_near + z_far)
    projective[2, 3] = (z_near * z_far)

    ndc = orthogonal(left=0, right=width, bottom=0, top=height, z_near=z_near, z_far=z_far)

    perspective = np.matmul(ndc, projective)

    return perspective, model_view