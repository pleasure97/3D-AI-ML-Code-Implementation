import os
import torch
import numpy as np
import imageio

from utils import get_ray_directions, get_rays_origin_and_direction, get_ndc_rays_origin_and_direction


def _minify(base_directory, factors=[], resolutions=[]):
    need2load = False
    for factor in factors:
        image_directory = os.path.join(base_directory, 'images_{}'.format(factor))
        if not os.path.exists(image_directory):
            need2load = True
    for resolution in resolutions:
        image_directory = os.path.join(base_directory, 'images_{}x{}'.format(resolution[1], resolution[0]))
        if not os.path.exists(image_directory):
            need2load = True
    if not need2load:
        return

    image_directory = os.path.join(base_directory, 'images')
    images = [os.path.join(image_directory, sub_directory) for sub_directory in sorted(os.listdir(image_directory))]
    images = [file for file in images if any([file.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    image_directory_orig = image_directory

    working_directory = os.getcwd()

    for fac_or_res in factors + resolutions:
        if isinstance(fac_or_res, int):
            name = 'images_{}'.format(fac_or_res)
            resize = '{}%'.format(100. / fac_or_res)
        else:
            name = 'images_{}x{}'.format(fac_or_res[1], fac_or_res[0])
            resize = '{}x{}'.format(fac_or_res[1], fac_or_res[0])
        image_directory = os.path.join(base_directory, name)
        if os.path.exists(image_directory):
            continue

        print('Minifying', fac_or_res, base_directory)

        os.makedirs(image_directory)

        extension = images[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resize, '-format', 'png', '*.{}'.format(extension)])
        print(args)
        os.chdir(image_directory)
        os.chdir(working_directory)

        if extension != 'png':
            print('Removed duplicates')
        print('Done')


def _load_data(base_directory, factor=None, width=None, height=None, load_images=True):
    poses_array = np.load(os.path.join(base_directory, 'poses_bound.npy'))
    poses = poses_array[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_array[:, -2:].transpose([1, 0])

    img0 = [os.path.join(base_directory, 'images', file) for file \
            in sorted(os.listdir(os.path.join(base_directory, 'images'))) \
            if file.endswith('JPG') or file.endswith('jpg') or file.endswith('png')][0]

    image_shape = imageio.v3.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(base_directory, factors=[factor])
        factor = factor
    elif height is not None:
        factor = image_shape[0] / float(height)
        width = int(image_shape[1] / factor)
        _minify(base_directory, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = image_shape[1] / float(width)
        height = int(image_shape[0] / factor)
        _minify(base_directory, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    image_directory = os.path.join(base_directory, 'images' + sfx)
    if not os.path.exists(image_directory):
        print(image_directory, 'does not exist')
        return

    image_files = [os.path.join(image_directory, file) for file in sorted(os.listdir(image_directory)) \
                   if file.endswith('JPG') or file.endswith('jpg') or file.endswith('png')]
    if poses.shape[-1] != len(image_files):
        print('Mismatch between images {} and poses {}'.format(len(image_files), poses.shape[-1]))
        return

    image_file_shape = imageio.v3.imread(image_files[0]).shape
    poses[:2, 4, :] = np.array(image_file_shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_images:
        return poses, bds

    def imread(file):
        if file.endswith('png'):
            return imageio.v3.imread(file, ignoregamma=True)
        else:
            return imageio.v3.imread(file)

    images = [imread(file)[..., :3] / 255. for file in image_files]
    images = np.stack(images, -1)

    print('Loaded image data', images.shape, poses[:, -1, 0])
    return poses, bds, images


def normalize(x):
    return x / np.linalg.norm(x)


def view_matrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def points2cam(points, c2w):
    tt = np.matmul(c2w[:3, :3].T, (points - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([view_matrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, radius, focal, z_delta, z_rate, rotations, N):
    render_poses = []
    radius = np.array(list(radius) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rotations, N + 1)[-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1.]) * radius)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([view_matrix(z, up, c), hwf], 1))

    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_

    return poses


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4])), [p.shape[0], 1, 1]], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_distance(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        point_min_distance = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return point_min_distance

    point_min_distance = min_line_distance(rays_o, rays_d)

    center = point_min_distance
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    radius = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / radius
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    radius *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radius_circle = np.sqrt(radius ** 2 - zh ** 2)
    new_poses = []

    for theta in np.linspace(0., 2. * np.pi, 120):
        cam_origin = np.array([radius_circle * np.cos(theta), radius_circle * np.sin(theta), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(cam_origin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = cam_origin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:, :3, :4], \
                                  np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_data(base_directory, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    poses, bds, images = _load_data(base_directory, factor=factor)
    print('Loaded', base_directory, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dimension to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        z_delta = close_depth * .2
        tt = poses[:, :3, 3]  # points2cam(poses[:3, 3, :].T, c2w).T
        radius = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        num_views = 120
        num_rotations = 2
        if path_zflat:
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            radius[2] = 0.
            num_rotations = 1
            num_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, radius, focal, z_delta, \
                                          z_rate=.5, rotations=num_rotations, N=num_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data :')
    print(poses.shape, images.shape, bds.shape)

    distances = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(distances)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    bounding_box = get_bbox3d(poses[:, :3, :4], poses[0, :3, -1], near=0., far=1.)

    return images, poses, bds, render_poses, i_test, bounding_box


def get_bbox3d(poses, hwf, near=0., far=1.):
    height, width, focal = hwf
    height, width = int(height), int(width)

    # ray directions in camera coordinates
    directions = get_ray_directions(height, width, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)

    for pose in poses:
        rays_o, rays_d = get_rays_origin_and_direction(directions, pose)
        rays_o, rays_d = get_ndc_rays_origin_and_direction(height, width, focal, 1., rays_o, rays_d)

        def find_min_max(point):
            for i in range(3):
                if min_bound[i] > point[i]:
                    min_bound[i] = point[i]
                if max_bound[i] < point[i]:
                    max_bound[i] = point[i]
            return

        for i in [0, width - 1, height * width - width, height * width - 1]:
            min_point = rays_o[i] + near * rays_d[i]
            max_point = rays_o[i] + far * rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound) - torch.tensor([0.1, 0.1, 0.0001]), \
            torch.tensor(max_bound) + torch.tensor([0.1, 0.1, 0.0001]))