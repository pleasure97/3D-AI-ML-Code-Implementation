import configargparse
import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tinycudann as tcnn
import json
import imageio
from tqdm import tqdm, trange

from utils import HashEncoder, get_multires_hash_encoding, hierarchical_sampling, \
                    get_rays, ndc_rays, get_rays_np, img2mse, mse2psnr, to8bit
from model import InstantNeRF
from load_data import load_data
from loss import total_variation_loss, sigma_sparsity_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify_network(network, batch_size):
    ''' Returns a network entered by the batch size. '''
    def batchify(embedding):
        return torch.cat([network(embedding[i: i + batch_size]) for i in range(0, embedding.shape(0), batch_size)], 0)

    return batchify


def run_network(images, rays_d, network, encoding, batch_size=2 ** 16):
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


def raw2outputs(raw, delta, rays_d, raw_noise_std=0., white_background=False):
    '''
    Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_coarse_samples along ray, 4]. Prediction from model.
        delta: [num_rays, num_coarse_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_coarse_samples]. Weights assigned to each sampled color.
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
    w_i = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,:-1]
    c_r = torch.sum(w_i[..., None] * c_i, -2)

    #

    depth_map = torch.sum(w_i * delta, -1) / torch.sum(w_i, -1)
    disparity_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    opacity_map = torch.sum(w_i, -1)

    #

    entropy = Categorical(probs=torch.cat([w_i, 1. - w_i.sum(-1, keepdims=True) + 1e-6], dim=-1)).entropy()

    return c_r, depth_map, disparity_map, opacity_map, w_i, entropy


def render_rays(r_batched, network, query, num_coarse_samples, embedding=None, include_raw=False, \
                perturb=0., num_fine_samples=0, fine_network=None, \
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
      num_coarse_samples : int. Number of different times to sample along each ray.
      include_raw : bool. If True, include model's raw, unprocessed predictions.
      perturb : float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      num_fine_samples : int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      fine_network : "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std : ...
      verbose : bool. If True, print more debugging info.
    Returns:
      rgb_map : [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map : [num_rays]. Disparity map. 1 / depth.
      acc_map : [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw : [num_rays, num_coarse_samples, 4]. Raw predictions from model.
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
    sample_intervals = torch.linspace(0., 1., num_coarse_samples)
    delta = near * (1. - sample_intervals) + far * (sample_intervals)
    delta = delta.expand([num_rays, num_coarse_samples])

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

    if num_fine_samples > 0:
        c_r_0, depth_map_0, opacity_map_0, entropy_0 = c_r, depth_map, opacity_map, entropy
        delta_mid = 0.5 * (delta[..., 1:] + delta[..., :-1])

        samples = hierarchical_sampling(delta_mid, weights[..., 1: -1], num_fine_samples, use_uniform=(perturb == 0.))
        samples = samples.detach()

        delta, _ = torch.sort(torch.cat([delta, samples], -1), -1)
        points = rays_o[..., None, :] + rays_d[..., None, :] * delta[..., :, None]

        network = network if fine_network is None else fine_network
        raw = query(points, viewdirs, network)
        c_r, disparity_map, opacity_map, weights, depth_map, entropy = raw2outputs(raw, delta, rays_d,
                                                                                       raw_noise_std, white_background)

    output_dict = {'c_r': c_r, 'depth_map': depth_map, 'opacity_map': opacity_map, 'entropy': entropy}

    if include_raw:
        output_dict['raw'] = raw

    if num_fine_samples > 0:
        output_dict['c_r_0'] = c_r_0
        output_dict['depth_map_0'] = depth_map_0
        output_dict['opacity_map_0'] = opacity_map_0
        output_dict['entropy'] = entropy
        output_dict['std'] = torch.std(samples, dim=-1, unbiased=False)

    return output_dict

def render(height, width, K, rays_per_memory = 2 ** 16, rays = None, cam2world = None, ndc = True,
           near = 0., far = 1., use_viewdirs = True, staticcam = None, **kwargs):
    '''
    Render rays
    Inputs:
      height: int. Height of image in pixels.
      width: int. Width of image in pixels.
      rays_per_memory: int. Maximum number of rays to process simultaneously.
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
    r_dict = batchify_rays(rays, rays_per_memory, **kwargs)

    for r in r_dict:
        r_shape = list(d_shape[:-1]) + list(r_dict[r].shape[1:])
        r_dict[r] = torch.reshape(r_dict[r], r_shape)

    r_extracted = ['c_r', 'depth_map', 'opacity_map']
    r_list = [r_dict[r] for r in r_extracted]
    r_dict = {r : r_dict[r] for r in r_dict if r not in r_extracted}

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

    output_channel = 5 if args.num_fine_samples > 0 else 4
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

    if args.num_fine_samples > 0 :
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
    base_directory = args.base_directory
    experiment_name = args.experiment_name

    # Load checkpoints
    if args.ckpt_path is not None:
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(base_directory, experiment_name, f) for f in sorted(os.listdir(os.path.join(base_directory, experiment_name))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        model.load_state_dict(ckpt['network_fn_state_dict'])

        fine_network.load_state_dict(ckpt['fine_network_state_dict'])

        embedded.load_state_dict(ckpt['embedded_state_dict'])

    train_kwargs = {
        'query' : query,
        'perturb' : args.perturb,
        'num_fine_samples' : args.num_fine_samples,
        'fine_network' : fine_network,
        'num_coarse_samples' : args.num_coarse_samples,
        'network_fn' : model,
        'embedded' : embedded,
        'use_viewdirs' : args.use_viewdirs,
        'white_background' : args.white_background,
        'raw_noise_std' : args.raw_noise_std,
    }

    test_kwargs = {t : train_kwargs[t] for t in train_kwargs}
    test_kwargs['perturb'] = False
    test_kwargs['raw_noise_std'] = 0.


    return train_kwargs, test_kwargs, start, model_params, optimizer


def config_parser():

    parser = configargparse.ArgumentParser()

    # Set the path required for the experiment
    parser.add_argument('--config', is_config_file = True, help = 'config file path')
    parser.add_argument('--base_directory', type = str, default = './checkpoints', help = 'where to store checkpoints')
    parser.add_argument('--experiment_name', type = str, help = 'the name of the experiment')
    parser.add_argument('--data_directory', type = str, default = './data/', help = 'path where input data is stored')

    # Set hyperparameters for training
    parser.add_argument('--num_layers', type = int, default = 8, help = 'layers in network')
    parser.add_argument('--num_channels', type = int, default = 256, help = 'channel per layer')
    parser.add_argument('--fine_num_layers', type = int, default = 8, help = 'layers in fine network')
    parser.add_argument('--fine_num_channels', type = int, default = 256, help = 'channel per layer in fine network')
    parser.add_argument('--finest_resolution', type = int, default = 512, help = 'finest resolution for hash embedding')
    parser.add_argument('--num_rays', type = int, default = 2 ** 12, help = 'number of rays per gradient step')
    parser.add_argument('--log2_hashmap_size', type = int, default = 19, help = 'log2 of hashmap size')
    parser.add_argument('--learning_rate', type = float, default = 5e-4, help = 'learning rate')
    parser.add_argument('--learning_rate_decay', type = int, default = 250, help = 'learning rate decay in 1000 steps')
    parser.add_argument('--rays_per_memory', type = int, default = 2 ** 15, help = 'number of rays processed in memory')
    parser.add_argument('--num_points', type = int, default = 2 ** 16, help = 'numer of points sent through network')
    parser.add_argument('--use_batching', action = 'store_true', help = 'only take random rays from one image at a time')
    parser.add_argument('--no_reload', action = 'store_true', help = 'do not reload weights from saved checkpoints')
    parser.add_argument('--coarse_npy', type = str, default = None, help = 'weights npy file for coarse network')

    # Set rendering options
    parser.add_argument('--num_coarse_samples', type = int, default = 64, help = 'number of coarse samples per ray')
    parser.add_argument('--num_fine_samples', type = int, default = 0, help = 'number of fine samples added per ray')
    parser.add_argument('--perturb', type = float, default = 1., help = 'set to 1. for jitter else 0.')
    parser.add_argument('--use_viewdirs', action = 'store_true', help = 'use full 5D input instead of 3D')
    parser.add_argument('--max_frequency_3D', type = int, default = 10, help = 'log2 of max frequency for 3D location')
    parser.add_argument('--max_frequency_2D', type = int, default = 4, help = 'log2 of max frequency for 2D direction')
    parser.add_argument('--raw_noise_std', type = float, default = 0., help = 'standard deviation of noise added')
    parser.add_argument('--render_only', action = 'store_true', help = 'render only render_poses path')
    parser.add_argument('--render_test', action = 'store_true', help = 'render the test set')
    parser.add_argument('--render_factor', type = int, default = 0, help = 'downsampling factor to speed up rendering')

    # Set options for cropping
    parser.add_argument('--crop_iteration', type = int, default = 0, help = 'number of steps to train on central crops')
    parser.add_argument('--crop_fraction', type = float, default = .5, help = 'fraction taken for central crops')

    # Set dataset options
    parser.add_argument('--half_resolution', action = 'store_true', help = 'load data at 400X400 instead of 800X800')
    parser.add_argument('--scene_id', type = str, default = 'scene0000_00', help = 'scene id to load from scannet')

    # Set logging and saving options
    parser.add_argument('--i_print', type = int, default = 100, help = 'frequency of console printout & metric loggin')
    parser.add_argument('--i_img', type = int, default = 500, help = 'frequency of tensorboard image logging')
    parser.add_argument('--i_weights', type = int, default = 10000, help = 'frequency of weights checkpoint saving')
    parser.add_argument('--i_test', type = int, default = 1000, help = 'frequency of testset saving')
    parser.add_argument('--i_video', type = int, default = 5000, help = 'frequency of render_poses video saving')
    parser.add_argument("--sparse_loss_weight", type = float, default = 1e-10, help = 'learning rate')
    parser.add_argument("--tv_loss_weight", type = float, default = 1e-6, help = 'learning rate')


    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()


    images, poses, render_poses, hwf, i_mode, bounding_box = \
        load_data(args.data_directory, args.scene_id, args.half_resolution)
    args.bounding_box = bounding_box
    print('Load dataset : images', images.shape, 'render_poses', render_poses.shape, \
          'height, width, focal', hwf, 'data_directory', args.data_directory)

    i_train, i_val, i_test = i_mode

    near, far = 0.1, 10.0

    # Cast intrinsics to right types
    height, width, focal = hwf
    height, width = int(height), int(width)
    hwf = [height, width, focal]

    K = np.array([
        [focal, 0, 0.5 * width],
        [0, focal, 0.5 * height],
        [0, 0, 1]
    ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log directory and copy the config file
    base_directory = args.base_directory
    args.experiment_name += "_fine" + str(args.finest_resolution) + "_hashmap_size" + str(args.log2_hashmap_size)
    args.experiment_name += "_lr" + str(args.learning_rate) + "_decay" + str(args.learning_rate_decay)
    args.experiment_name += "_Adam"
    if args.sparse_loss_weight > 0:
        args.experiment_name += "_sparse" + str(args.sparse_loss_weight)
    args.experiment_name += "_TV" + str(args.tv_loss_weight)
    experiment_name = args.experiment_name

    os.makedirs(os.path.join(base_directory, experiment_name), exist_ok = True)
    file = os.path.join(base_directory, experiment_name, 'args.txt')
    with open(file, 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {} \n".format(arg, attr))
    if args.config is not None:
        file = os.path.join(base_directory, experiment_name, 'config.txt')
        with open(file, 'w') as f:
            f.write(open(args.config, 'r').read())

    # Create NeRF model
    train_kwargs, test_kwargs, start, model_params, optimizer = create_NeRF(args)
    global_step = start

    bds_dict = {"near" : near, "far" : far}

    train_kwargs.update(bds_dict)
    test_kwargs.update(bds_dict)

    # Move test data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        with torch.no_grad():
            if args.render_test:
                images = images[i_test]
            else:
                images = None

            test_save_directory = os.path.join(base_directory, experiment_name, \
                                               'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path'), \
                                               start)
            os.makedirs(test_save_directory, exist_ok = True)
            print('test poses : ', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.rays_per_memory, test_kwargs, \
                                  gt_imgs = images, savedir = test_save_directory, render_factor = args.render_factor)
            print('Done rendering', test_save_directory)
            imageio.mimwrite(os.path.join(test_save_directory, 'video.mp4'), \
                             255 * np.clip(rgbs, 0, 1).astype(np.uint8), \
                             fps = 30, quality = 8)

            return

    # Prepare ray batch tensor if batching random rays
    num_rays = args.num_rays
    use_batching = args.use_batching
    if use_batching:
        print('Get rays.')
        rays = np.stack([get_rays_np(height, width, K, p) for p in poses[:, :3, :4]], 0)
        print('Concatenate rays.')
        rays_rgb = np.concatenate([rays, images[:, None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]) # [(N-1) * H * W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('Shuffle rays.')
        np.random.shuffle(rays_rgb)
        print('Done.')
        i_batch = 0

    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    num_iteration = 50000
    print('Start')
    print('TRAIN Views are', i_train)
    print('TEST Views are', i_test)
    print('VAL Views are', i_val)

    losses = []
    psnrs = []
    times = []

    start += 1
    tic = time.time()
    for i in trange(start, num_iteration + 1):
        # Sample random ray batch
        if use_batching:
            batch = rays_rgb[i_batch : i_batch + num_rays] # [B, 2+1, 3 * ?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += num_rays
            if i_batch >= rays_rgb.shape[0]:
                print('Shuffle data after an epoch')
                idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[idx]
                i_batch = 0

        else:
            # Random from one image
            image_idx = np.random.choice(i_train)
            target = images[image_idx]
            target = torch.Tensor(target).to(device)
            pose = poses[image_idx, :3, :4]

            if num_rays is not None:
                rays_o, rays_d = get_rays(height, width, K, torch.Tensor(pose))

                if i < args.crop_iteration:
                    dh = int(height // 2 * args.crop_fraction)
                    dw = int(width // 2 * args.crop_fraction)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(height//2 - dh, height//2 + dh - 1, 2 * dh),
                            torch.linspace(width//2 - dw, width//2 + dw - 1, 2 * dw)
                        ), -1)

                    if i == start:
                        print(f"Center cropping of size {2*dh} X {2*dw} is enabled until iter {args.crop_iteration}")

                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, height - 1, height),
                            torch.linspace(0, width - 1, width)
                        ), -1)

                coords = torch.reshape(coords, [-1, 2]) # (H * W, 2)
                sampled_indices = np.random.choice(coords.shape[0], size = [num_rays], replace = False)
                sampled_coords = coords[sampled_indices].long()
                rays_o = rays_o[sampled_coords[:, 0], sampled_coords[:, 1]]
                rays_d = rays_d[sampled_coords[:, 0], sampled_coords[:, 1]]
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[sampled_coords[:, 0], sampled_coords[:, 1]]

            # Optimization loop
            rgb, depth, opacity, extras = render(height, width, K, \
                                                 rays_per_memory = args.rays_per_memory, rays = batch_rays, \
                                                 verbose = i < 10,use_viewdirs = True, \
                                                 **train_kwargs)

            optimizer.zero_grad()
            image_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = image_loss
            psnr = mse2psnr(image_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

            sparsity_loss = args.sparse_loss_weight * (extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
            loss += sparsity_loss

            num_levels = train_kwargs["embedded"].num_levels
            min_resolution = train_kwargs["embedded"].base_resolution
            max_resolution = train_kwargs["embedded"].finest_resolution
            log2_hashmap_size = train_kwargs["embedded"].log2_hashmap_size
            tv_loss = sum(total_variation_loss(train_kwargs["embedded"].embeddings[i], \
                                               min_resolution, max_resolution, \
                                               i, log2_hashmap_size, \
                                               num_levels = num_levels) for i in range(num_levels))
            loss += args.tv_loss_weight * tv_loss
            if i > 1000 :
                args.tv_loss_weight = 0.

        loss.backward()
        optimizer.step()

        # Update learning rate
        decay_rate = 0.1
        decay_steps = args.learning_rate_decay * 1000
        new_learning_rate = args.learning_rate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['learning_rate'] = new_learning_rate

        toc = time.time()
        t = toc - tic

        if i % args.i_weights == 0:
            path = os.path.join(base_directory, experiment_name, '{:06d}.tar'.format(i))
            torch.save({
                'global_step' : global_step,
                'network_fn_state_dict' : train_kwargs['network_fn'].state_dict(),
                'fine_network_state_dict' : train_kwargs['fine_network'].state_dict(),
                'embedded_state_dict' : train_kwargs['embedded'].state_dict(),
                'optimizer.state_dict' : optimizer.state_dict(),
            }, path)

        print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0 :
            with torch.no_grad():
                rgbs, depths = render_path(render_poses, hwf, K, args.rays_per_memory, test_kwargs)
            print('Saving rgbs :', rgbs.shape, 'and depths :', depths.shape)
            moviebase = os.path.join(base_directory, experiment_name, '{}_spiral_{:06d}_'.format(experiment_name, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8bit(rgbs), fps = 30, quality = 8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8bit(depths / np.max(depths)), fps = 30, quality = 8)

        if i % args.i_test == 0 and i > 0:
            test_save_directory = os.path.join(base_directory, experiment_name, 'testset_{:06d}'.format(i))
            os.makedirs(test_save_directory, exist_ok = True)
            print('test poses shape :', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.rays_per_memory, test_kwargs, \
                            gt_imgs = images[i_test], savedir = test_save_directory)
            print('Save test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iteration : {i} Loss : {loss.item()} PSNR : {psnr.item()}")
            losses.append(loss.item())
            psnrs.append(psnr.item())
            times.append(t)
            loss_psnr_time = {"losses" : losses, "psnrs" : psnrs, "time" : times}

            with open(os.path.join(base_directory, experiment_name, "loss_vs_time.pkl", "wb")) as fp:
                pickle.dump(loss_psnr_time, fp)

        global_step += 1

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
