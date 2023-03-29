import torch
import numpy as np
import os
import json
import imageio
import cv2
import pyvista as pv


def translate(t):
    translation = torch.Tensor([1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, t],
                               [0, 0, 0, 1], ).float()

    return translation


def rotate_phi(phi):
    rotation = torch.Tensor([1, 0, 0, 0],
                            [0, np.cos(phi), -np.sin(phi), 0],
                            [0, np.sin(phi), np.cos(phi), 0],
                            [0, 0, 0, 1]).float()

    return rotation


def rotate_theta(theta):
    rotation = torch.Tensor([np.cos(theta), 0, -np.sin(theta), 0],
                            [0, 1, 0, 0],
                            [np.sin(theta), 0, np.cos(theta), 0],
                            [0, 0, 0, 1]).float()

    return rotation


def pose_spherical(theta, phi, radius):
    c2w = translate(radius)
    c2w = rotate_phi(phi / 180. * np.pi) @ c2w
    c2w = rotate_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    return c2w


def load_data(base_directory, scene_id, half_resolution=False, train_skip=10, test_skip=1):
    scan_directory = os.path.join(base_directory, "scans")
    base_directory = os.path.joiin(base_directory, "nerfstyle_" + scene_id)

    modes = ['train', 'val', 'test']
    metas = {}

    for mode in modes:
        with open(os.path.join(base_directory, 'transforms_{}.json'.format(mode)), 'r') as file:
            metas[mode] = json.load(file)

    images_list = []
    poses_list = []
    counts = [0]

    for mode in modes:
        meta = metas[mode]
        images = []
        poses = []
        if mode == "train":
            skip = train_skip
        else:
            skip = test_skip

        for frame in meta['frames'][::skip]:
            file_name = os.path.join(base_directory, frame['file_path'] + '.png')
            images.append(imageio.imread(file_name))
            pose = np.array(frame['transform_matrix'])

            # ScanNet uses OpenCV convention
            poses[:3, 1] *= -1
            poses[:3, 2] *= -1

            poses.append(pose)

        images = (np.array(images) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + images.shape[0])
        images_list.append(images)
        poses_list.append(poses)

    i_mode = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    images = np.concatenate(images_list, 0)
    poses = np.concatenate(poses_list, 0)

    height, width = images[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30., 4.) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_resolution:
        height //= 2
        width //= 2
        focal /= 2.

        half_resolution_images = np.zeros((images.shape[0], height, width, 3))
        for i, image in enumerate(images):
            half_resolution_images[i] = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        images = half_resolution_images


    # load scene mesh
    mesh = pv.read(os.path.join(scan_directory, scene_id, f"{scene_id}_vh_clean.ply"))

    # get the bounding box
    bounding_box = torch.tensor(mesh.bounds[::2]) - 1, torch.tensor(mesh.bounds[1::2]) + 1

    return images, poses, render_poses, [height, width, focal], i_mode, bounding_box

