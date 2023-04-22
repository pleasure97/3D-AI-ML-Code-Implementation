import os
import configargparse
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import tqdm
import multiprocessing as mp
from functools import partial
from preprocess import download_video, trim_and_crop
from torch.utils.tensorboard import SummaryWriter
from dataset import FramesDataset, DatasetRepeater
from utils import load_config


def train(experiment_name):
    config = load_config(experiment_name)

    model_params = config["model_params"]
    model_name = model_params.model_name

    checkpoint_dir = f'{config.root_directory}/checkpoints/{experiment_name}'
    tensorboard_dir = f'{config.root_director}/tensorboard/{model_name}/{experiment_name}'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    logger = SummaryWriter(log_dir=tensorboard_dir)

    scaler = torch.cuda.amp.GradScaler()

    train_params = config["train_params"]
    train_dataset = DatasetRepeater(FramesDataset(), num_repeats=100)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              sampler=train_sampler)





def evaluate():
    pass


def predict():
    pass


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, required=True,
                        help='List of youtube video ids')
    parser.add_argument('--dataset_dir', type=str, default='data/youtube_videos',
                        help='Location to download videos')
    parser.add_argument('--clip_dir', type=str, required=True,
                        help='Dir containing youtube clips.')
    parser.add_argument('--clip_info_file', type=str, required=True,
                        help='File containing clip information.')
    parser.add_argument('--num_mp_workers', type=int, default=8,
                        help='Number of multiprocessing workers')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of GPU workers related to CUDA')

    args = parser.parse_args()
    action = args.action

    video_ids = []
    with open(args.data_list) as fin:
        for line in fin:
            video_ids.append(line.strip())

    os.makedirs(args.dataset_dir, exist_ok=True)

    downloader = partial(download_video, args.dataset_dir)

    start = time.time()
    pool_size = args.num_mp_workers
    print(f'Using pool size of {pool_size}')

    with mp.Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(downloader, video_ids), total=len(video_ids)))
    print('Elapsed time : {:2f}'.format(time.time() - start))

    clip_info = []
    with open(args.clip_info_file) as fin:
        for line in fin:
            clip_info.append(line.strip())

    # Create output folder.
    os.makedirs(args.output_dir, exist_ok=True)

    # Download videos.
    downloader = partial(trim_and_crop, args.clip_dir, args.output_dir)

    start = time.time()
    pool_size = args.num_mp_workers
    print(f'Using pool size of {pool_size}')
    with mp.Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(downloader, clip_info), total=len(clip_info)))
    print('Elapsed time: %.2f' % (time.time() - start))

    if action == "train":
        train(

        )

    if action == "evaluate":
        evaluate(

        )

    if action == "predict":
        predict(

        )
