import os
import numpy as np
import glob
from skimage import io
from skimage.color import gray2rgb
from skimage.util import img_as_float32
from sklearn.model_selection import train_test_split
from imageio import mimread
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform


def load_video(name, frame_shape):
    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array([img_as_float32(io.imread(os.path.join(name, str(frames[idx], encoding='utf-8'))))
                                for idx in range(num_frames)])
    elif name.lower().endswith(".gif") or name.lower().endswith(".mp4"):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception(f"Unknown file extensions {name}")


class FramesDataset(Dataset):

    def __init__(
            self,
            root_directory='',
            frame_shape=(256, 256, 3),
            sample_id=True,
            is_train=True,
            random_seed=0,
            pairs_list=None,
            augmentation_params=None):
        self.root_directory = root_directory
        self.videos = os.listdir(root_directory)
        self.frame_shape = tuple(frame_shape)
        self.sample_id = sample_id
        self.pairs_list = pairs_list
        self.is_train = is_train

        if augmentation_params is None:
            augmentation_params = \
                {"flip_param": {"horizontal_flip": True, "time_flip": True},
                 "jitter_param": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.1}}

        if os.path.exists(os.path.join(root_directory, "train")):
            if sample_id:
                train_videos = {os.path.basename(video).split("#")[0]
                                for video in os.listdir(os.path.join(root_directory, "train"))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_directory, "train"))
            test_videos = os.listdir(os.path.join(root_directory, "test"))
            self.root_directory = os.path.join(self.root_directory, "train" if is_train else "test")
        else:
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.videos = test_videos
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.sample_id:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_directory, name + "*.mp4")))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_directory, name)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, str(frames[idx], encoding='utf-8'))))
                           for idx in frame_idx]
        else:
            video_array = load_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) \
                if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        if self.is_train:
            source = np.array(video_array[0], dtype="float32")
            driving = np.array(video_array[1], dtype="float32")

            source = source.transpose((2, 0, 1))
            driving = driving.transpose((2, 0, 1))

            return source, driving

        else:
            video = np.array(video_array, dtype="float32")
            video = video.transpose((3, 0, 1, 2))

            return video


class DatasetRepeater(Dataset):

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats + self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
