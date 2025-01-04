import torch.cuda


def use_gpu(network, init_type='normal', init_gain=.02, gpu_ids=[]):
    # Use gpu if it's available
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        network.to(gpu_ids[0])
        network = torch.nn.DataParallel(network, gpu_ids)

    # Initialize weights of the network
    classname = network.__class__.__name__
    if hasattr(network, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(network.weight.data, 0., init_gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(network.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(network.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(network.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(f'Initialization method {init_type} is not implemented.')
        if hasattr(network, 'bias') and network.bias is not None:
            torch.nn.init.constant_(network.bias.data, 0.)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(network.weight.data, 1., init_gain)
        torch.nn.init.constant_(network.bias.data, 0.)

    print(f'Initialize network with {init_type}')

    return network
