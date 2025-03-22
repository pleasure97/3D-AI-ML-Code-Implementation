from loss import get_denoising_loss
total_iters = 100_000
warmup_iters = 2_000

current_iter = 0

is_object =

point_distribution_loss =
denoising_distribution_loss = get_denoising_loss()
novel_view_loss =
loss = torch.where(current_iter > warmup_iters, denoising_distribution_loss + novel_view_loss, point_distribution_loss * torch.where(is_object, 1, 0))
loss.backward()
