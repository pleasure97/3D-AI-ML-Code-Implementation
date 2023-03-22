import configargparse

def batchify(network, images, batch_size):
    '''
    Returns a network entered by the batch size.      
    '''
    assert batch_size is not None
    
    return torch.cat([network(images[i : i + batch_size]) for i in range(0, images.shape(0), batch_size)], 0)


def run_network(images, rays_d, network, encoding = get_multires_hash_encoding(args), batch_size = 2 ** 16):
    
    images_flattened = torch.reshape(images, [-1, images.shape[-1]])
    
    embedded, out_dim = encoding(images_flattened)
    
    d_expanded = rays_d[:, None].expand(images.shape)
    
    d_flattened = 
    
    
    
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
    