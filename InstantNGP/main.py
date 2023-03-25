import configargparse
import torch
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tcnn
import json

from utils import get_multires_hash_encoding, get_rays, ndc_rays
from encoder import HashEncoder
from model import InstantNeRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify_network(network, batch_size):
    ''' Returns a network entered by the batch size. '''
    def batchify(embedding):
        return torch.cat([network(embedding[i: i + batch_size]) for i in range(0, embedding.shape(0), batch_size)], 0)

    return batchify


def run_network(images, rays_d, network, encoding=get_multires_hash_encoding(args), batch_size=2 ** 16):
    images_flattened = torch.reshape(images, [-1, images.shape[-1]])
    embedded, out_dim = encoding(images_flattened)

    #
    d_expanded = rays_d[:, None].expand(images.shape)

    #
    d_flattened = torch.reshape(d_expanded, [-1, d_expanded.shape[-1]])

    #
    d_embedded = torch.cat([embedded, d_flattened], -1)
    network_output = batchify_network(network, batch_size)(d_embedded)
    network_output[~out_dim, -1] = 0

    output = torch.reshape(network_output, list(images.shape[:-1]) + [network_output.shape[-1]])

    return output


def batchify_rays(r_flattened, batch_size=2 ** 16, **kwargs):
    '''
    Render rays in smaller minibatches to avoid out of memory.
    '''

    r_dict = {}

    for i in range(0, r_flattened.shape[0], batch_size):
        rays = render_rays(r_flattened[i: i + batch_size], **kwargs)
        for ray in rays:
            if ray not in r_dict:
                r_dict[ray] = []
            r_dict[ray].append(rays[ray])

    r_dict = {r: torch.cat(r_dict[r], 0) for r in r_dict}

    return r_dict


def raw2outputs(raw, delta, raw_noise_std=0., white_background=False):
    '''
    Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        delta: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    '''
    from torch.distributions import Categorical

    #

    dists = delta[..., 1:] - delta[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    dists *= torch.norm(rays_d[..., None, :], dim=-1)

    #

    c_i = torch.sigmoid(raw[..., :3])

    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    else:
        noise = 0.

    def raw2alpha(raw, dists, activation=F.relu):
        return 1. - torch.exp(-activation(raw) * dists)

    #

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    w_i = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:.:-1]
    c_r = torch.sum(w_i[..., None] * c_i, -2)

    #

    depth_map = torch.sum(w_i * delta, -1) / torch.sum(w_i, -1)
    disparity_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    opacity_map = torch.sum(weights, -1)

    #

    entropy = Categorical(probs=torch.cat([w_i, 1. - w_i.sum(-1, keepdims=True) + 1e-6], dim=-1).entropy()

    return c_r, depth_map, disparity_map, opacity_map, weights, entropy


def render_rays(r_batched, network, query, num_samples, embedding=None, include_raw=False, \
                perturb=0., num_importance=0, fine_network=None, \
                white_background=True, raw_noise_std=0., verbose=False):
    '''
    Volumetric rendering.
    Args:
      r_batched : array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network: function. Model for predicting RGB and density at each point
        in space.
      query : function used for passing queries to network_fn.
      num_samples : int. Number of different times to sample along each ray.
      include_raw : bool. If True, include model's raw, unprocessed predictions.
      perturb : float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      num_importance : int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      fine_network : "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std : ...
      verbose : bool. If True, print more debugging info.
    Returns:
      rgb_map : [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map : [num_rays]. Disparity map. 1 / depth.
      acc_map : [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw : [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0 : See rgb_map. Output for coarse model.
      disp0 : See disp_map. Output for coarse model.
      acc0 : See acc_map. Output for coarse model.
      z_std : [num_rays]. Standard deviation of distances along ray for each sample.
    '''

    #
    num_rays = r_batched.shape[0]
    rays_o, rays_d = r_batched[:, :3], r_batched[:, 3:6]

    #
    viewdirs = r_batched[:, -3:] if r_batched.shape[-1] > 8 else None
    bounds = torch.reshape(r_batched[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    #
    sample_intervals = torch.linspace(0., 1., num_samples)
    delta = near * (1. - sample_intervals) + far * (sample_intervals)
    delta = delta.expand([num_rays, num_samples])

    #
    if perturb > 0.:
        mid = .5 * (delta[..., 1:] + delta[..., :-1])
        upper = torch.cat([mid, delta[-1:]], -1)
        lower = torch.cat([delta[..., :1], mid], -1)
        stratified_samples = torch.rand(delta.shape)
        delta = lower + (upper - lower) * stratified_samples

    #

    points = rays_o[..., None, :] + rays_d[..., None, :] * delta[..., :, None]

    #

    raw = query(points, viewdirs, network)
    c_r, disparity_map, opacity_map, weights, depth_map, entropy = raw2outputs(raw, delta, rays_d, raw_noise_std,
                                                                                   white_background)

    #

    if num_importance > 0:
        c_r_0, depth_map_0, opacity_map_0, entropy_0 = c_r, depth_map, opacity_map, entropy
        delta_mid = 0.5 * (delta[..., 1:] + delta[..., :-1])

        samples = hierarchical_sampling(delta_mid, weights[..., 1: -1], num_importance, use_uniform=(perturb == 0.))
        samples = samples.detach()

        delta, _ = torch.sort(torch.cat([delta, z_samples], -1), -1)
        points = rays_o[..., None, :] + rays_d[..., None, :] * delta[..., :, None]

        network = network_coarse if network_fine is None else network_fine
        raw = query(points, viewdirs, network)
        c_r, disparity_map, opacity_map, weights, depth_map, entropy = raw2outputs(raw, delta, rays_d,
                                                                                       raw_noise_std, white_background)

    output_dict = {'c_r': c_r, 'depth_map': depth_map, 'opacity_map': opacity_map, 'entropy': entropy}

    if include_raw:
        output_dict['raw'] = raw

    if num_importance > 0:
        output_dict['c_r_0'] = c_r_0
        output_dict['depth_map_0'] = depth_map_0
        output_dict['opacity_map_0'] = opacity_map_0
        output_dict['entropy'] = entropy
        output_dict['std'] = torch.std(samples, dim=-1, unbiased=False)

    return output_dict

def render(height, width, K, batch_size = 2 ** 16, rays = None, cam2world = None, ndc = True,
           near = 0., far = 1., use_viewdirs = True, staticcam = None, **kwargs):
    '''
    Render rays
    Inputs:
      height: int. Height of image in pixels.
      width: int. Width of image in pixels.
      batch_size: int. Maximum number of rays to process simultaneously.
        Used to control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3].
        Ray origin and direction for each example in batch.
      cam2world: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      staticcam: array of shape [3, 4].
      If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
    Outputs:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disparity_map: [batch_size]. Disparity map. Inverse of depth.
      opacity_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    '''

    if cam2world is not None:
        rays_o, rays_d = get_rays(height, width, K, cam2world)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if staticcam is not None:
            rays_o, rays_d = get_rays(height, width, K, staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim = -1, keepdim = True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    d_shape = rays_d.shape

    if ndc:
        rays_o, rays_d = ndc_rays(height, width, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    r_dict = batchify_rays(rays, batch_size, **kwargs)

    for r in r_dict:
        r_shape = list(d_shape[:-1]) + list(r_dict[r].shape[1:])
        r_dict[r] = torch.reshape(r_dict[r], r_shape)

    r_extracted = ['c_r', 'depth_map', 'opacity_map']
    r_list = [r_dict[r] for r in r_extracted]
    r_dict = {r : r_dict[r] for r in r_dict if r not in r_extract}

    return r_list + [r_dict]


def render_path(render_poses, hwf, K, batch_size, render_kwargs, gt_imgs = None, savedir = None, render_factor = 0):

    height, width, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']

    if render_factor != 0:
        height //= render_factor
        width //= render_factor
        focal /= render_factor

    rgbs, depths, psnrs = [], [], []

    t = time.time()
    for i, cam2world in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        color, depth, opacity, _ = render(height, width, K,
                                          batch_size = batch_size, cam2world = cam2world[:3, :4],
                                          **render_kwargs)
        rgbs.append(color.cpu().numpy())

        # normalize depth to [0, 1]
        depth = (depth - near) / (far - near)
        depths.append(depth.cpu().numpy())

        if gt_imgs is not None and render_factor == 0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = - 10. * np.log10(np.mean(np.square(color.cpu().numpy() - gt_img)))
            psnrs.append(p)

        if savedir is not None:
            fig = plt.figure(figsize = (25, 15))
            ax = fig.add_subplot(1, 2, 1)
            color8 = (255 * np.clip(color[-1], 0, 1)).astype(np.uint8)
            ax.imshow(color8)
            ax.axis('off')
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(depths[-1], cmap = 'plasma', vmin = 0, vmax = 1)
            ax.axis('off')

            filename = os.path.join(savedir, '{:03d}.png'.format(i))

            plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)


    rgbs = torch.stack(rgbs, 0)
    depths = torch.stack(depths, 0)

    if gt_imgs is not None and render_factor == 0:
        avg_psnr = sum(psnrs) / len(psnrs)
        print("Average PNSR over test set : ", avg_psnr)
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)

    return rgbs, depths

def create_NeRF(args):

    embedded, input_channel = get_multires_hash_encoding(args.multires, encoder = HashEncoder)
    embedding_params = list(embedded.parameters())

    input_channel_views = 0
    embeddirs = None

    output_channel = 5 if args.num_importance > 0 else 4
    skips = [4]

    if args.use_tcnn :
        with open('./config_hash.json') as f:
            config = json.load(f)
        model = tcnn.NetworkWithInputEncoding(
            config.n_input_dims, config.n_output_dims,
            config["encoding"], config["network"])
    else:
        model = InstantNeRF(num_layers = 2,
                            hidden_dim = 64,
                            geo_feat_dim = 15,
                            num_layers_color = 3,
                            hidden_dim_color = 64,
                            input_channel = input_channel,
                            input_channel_views = input_channel_views
                            ).to(device)

    model_params = list(model.parameters())

    fine_network = None

    if args.num_importance > 0 :
        fine_network = InstantNeRF(num_layers = 2,
                                   hidden_dim = 64,
                                   geo_feat_dim = 15,
                                   num_layers_color = 3,
                                   hidden_dim_color = 64,
                                   input_channel = input_channel,
                                   input_channel_views = input_channel_views).to(device)

        model_params += list(fine_network.parameters())

    query = lambda images, rays_d, network : run_network(images, rays_d, network,
                                                         encoding = get_multires_hash_encoding(args),
                                                         batch_size = args.batch_size)

    optimizer = torch.optim.Adam(params = model_params, lr = args.lr, betas = (0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ckpt_path is not None:
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    ## scannet flags
    parser.add_argument("--scannet_sceneID", type=str, default='scene0000_00',
                        help='sceneID to load from scannet')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')

    return parser

def train():

    parser = config_parser()

    args = parser.parse_arg()

    if args.dataset == 'llff':
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(args.data_dir,
                                                                                args.factor,
                                                                                recenter = True,
                                                                                bd_factor = 0.75,
                                                                                spherify = args.shperify
                                                                                )

def main():
    # 모델 및 최적화 알고리즘 정의
    model = MyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 체크포인트 저장 경로 설정
    checkpoint_dir = 'checkpoints/'

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터 배치를 가져옵니다.
            data, target = data.to(device), target.to(device)

            # 모델을 통해 예측값을 계산합니다.
            output = model(data)

            # Loss function을 사용하여 예측값과 실제값 간의 오차를 계산합니다.
            loss = criterion(output, target)

            # 이전 배치에서의 누적된 그라디언트를 지웁니다.
            optimizer.zero_grad()

            # Loss의 그라디언트를 계산합니다.
            loss.backward()

            # Optimizer를 사용하여 파라미터를 업데이트합니다.
            optimizer.step()

            # 일부 배치마다 손실을 출력합니다.
            if batch_idx % log_interval == 0:

            # 일정 주기마다 체크포인트 저장
            if epoch % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir + f'model_epoch{epoch}_acc{accuracy}.pt'
                torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
