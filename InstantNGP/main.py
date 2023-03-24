import configargparse

def batchify_network(network, batch_size):
    '''
    Returns a network entered by the batch size.      
    '''
    assert batch_size is not None
    
    def batchify(embedding):
        
        return torch.cat([network(embedding[i : i + batch_size]) for i in range(0, embedding.shape(0), batch_size)], 0)

    return batchify



def run_network(images, rays_d, network, encoding = get_multires_hash_encoding(args), batch_size = 2 ** 16):
    
    images_flattened = torch.reshape(images, [-1, images.shape[-1]])
    
    embedded, out_dim = encoding(images_flattened)
    
    #
    
    d_expanded = rays_d[:, None].expand(images.shape)
    
    #
    
    d_flattened = torch.reshape(d_expanded, [-1, d_expanded.shape[-1]])
    
    #
    
    d_embedded = torch.cat([embedded, d_flattened], -1)
    
    network_output = batchify_network(network, batch_size)(d_embedded)
    
    network_output[~out_dim,-1] = 0
    
    output = torch.reshape(network_output, list(images.shape[:-1]) + [network_output.shape[-1]])
    
    return output


    
def batchify_rays(r_flattened, batch_size = 2 ** 16, **kwargs):
    
    '''
    Render rays in smaller minibatches to avoid out of memory.
    '''
    
    r_dict = {}
    
    for i in range(0, r_flattened.shape[0], batch_size):
        
        rays = render_rays(r_flattend[i : i + batch_size], **kwargs)
        
        for ray in rays:
            
            if ray not in r_dict:
                
                r_dict[ray] = []
             
            r_dict[ray].append(rays[ray])
     
    r_dict = {r : torch.cat(ray_dict[r], 0) for r in r_dict}
        
    return r_dict

def raw2outputs(raw, delta, raw_noise_std = 0., white_background = False):
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

    #
    
    dists = delta[..., 1:] - delta[..., :-1]
    
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    
    dists *= torch.norm(rays_d[..., None, :], dim = -1)
    
    #
    
    c_i = torch.sigmoid(raw[..., :3])
    
    if raw_noise_std > 0.:
        
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        
    else :
        
        noise = 0.
        
    def raw2alpha(raw, dists, activation = F.relu):
        
        return 1.- torch.exp(-activation(raw) * dists)
    
    #
    
    alpha = raw2alpha(raw[..., 3] + noise, dists)
    
    w_i = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10],-1),-1)[:. :-1]
    
    c_r = torch.sum(w_i[..., None] * c_i, -2)
    
    #
    
    depth_map = torch.sum(w_i * delta, -1) / torch.sum(w_i, -1)
    
    disparity_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    
    accumulated_map = torch.sum(weights, -1)
    
    #
    
    entropy = Categorical(probs = torch.cat([w_i, 1. - w_i.sum(-1, keepdims = True) + 1e-6], dim = -1).entropy()

def render_rays(r_batched, network, query, num_samples, embedding = None, include_raw = False, \
                perturb = 0., num_importance = 0, fine_network = None, \
                white_background = True ,raw_noise_std = 0., verbose = False):
    
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
    near, far = bounds[..., 0], bounds[...,1]
    
    #
    sample_intervals = torch.linspace(0., 1., num_samples)
    
    delta =  near * (1. - sample_intervals) + far * (sample_intervals)
    
    delta = delta.expand([num_rays, num_samples])
    
    #
    if perturb > 0. :
        
        mid = .5 * (delta[..., 1:] + delta[..., :-1])
        upper = torch.cat([mid, delta[-1:]], -1)
        lower = torch.cat([delta[..., :1], mid], -1)
        
        stratified_samples = torch.rand(delta.shape)
        
        delta = lower + (upper - lower) * stratified_samples
        
    #
    
    points = rays_o[..., None, :] + rays_d[..., None, :] * delta[..., :, None]
    
    #
    
    raw = query(points, viewdirs, network)
    
    
    

    
    
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
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')

    return parser


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
    