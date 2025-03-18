import torch

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int=10_000):
    """
    Sinusodial Timestep Embeddings. 
    Args:
      timesteps: torch.Tensor. 1D tensor of shape [N] containing timestep values. 
      dim: int. Dimension of the Output Embeddings. 
      max_period: int. Minimum Frequency of the Embeddings. 

    Returns:
      embedding: torch.Tensor. [N * dim] tensor of timestep embeddings. 
    """
    # Assign half of the embedding dimension to sine and cosine respectively. 
    half = dim // 2
    # Determine the minimum frequency 
    max_period_log = torch.log(torch.tensor(max_period, dtype=timesteps.dtype, device=timesteps.device))
    # Generate the frequency values 
    arange = torch.arange(half, dtype=timesteps.dtype, device=timesteps.device)
    frequencys = torch.exp(-max_period_log * arange / half)
    
    # timesteps: [N] -> [N, 1] / frequencys: [half] -> [1, half]
    args = timesteps.unsqueeze(1) * frequencys.unsqueeze(0) # [N, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
      zeros = torch.zeros(timesteps.shape[0], 1, dtype=timesteps.dtype, device=timesteps.device)
      embedding = torch.cat([embedding, zeros], dim=-1)

    return embedding
