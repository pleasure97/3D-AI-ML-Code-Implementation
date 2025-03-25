import torch 

def get_point_distribution_loss(decoder: GaussianDecoder, rays_o, rays_d, k:int, sigma_0: float=0.5):
  assert isinstance(decoder, GaussianDecoder), f'Expected Type is GaussianDecoder, but Current Type is {decoder.type}'

  l_t_list = []
  
  for i in range(k):
    u_t = decoder.weight[k] * decoder.u_near + (1 - decoder.weight[k]) * decoder.u_far
    l_t = torch.abs(u_t * rays_d[k])
    l_t_list.append(l_t)
  
  l_t_tensor = torch.stack(l_t_list)

  mean_l_t = torch.mean(l_t_tensor)
  var_l_t = torch.var(l_t_tensor, unbiased=False)
  std_l_t = torch.sqrt(var_l_t)

  abs_o_list = [torch.abs(ray_o) for ray_o in rays_o]
  mean_abs_o = torch.mean(torch.stack(abs_o_list))

  point_distribution_loss = torch.mean(l_t_tensor - ((l_t_tensor - mean_l_t) / std_l_t * sigma_0 + mean_abs_o))

  return point_distribution_loss
