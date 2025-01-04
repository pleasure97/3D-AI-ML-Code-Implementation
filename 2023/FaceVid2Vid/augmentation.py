import random
import warnings

import PIL.Image
import numpy as np
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import adjust_brightness, adjust_saturation, adjust_hue, adjust_contrast
from skimage.util import img_as_ubyte, img_as_float


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            return [np.fliplr(c) for c in clip]

        return clip


class ColorJitter(object):

    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None

        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            brightness, contrast, saturation, hue \
                = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: adjust_brightness(img, brightness))
            if contrast is not None:
                img_transforms.append(lambda img: adjust_contrast(img, contrast))
            if saturation is not None:
                img_transforms.append(lambda img: adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: adjust_hue(img, hue))

            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, ToPILImage()] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for c in clip:
                    jittered_img = c
                    for img_transform in img_transforms:
                        jittered_img = img_transform(jittered_img)
                    jittered_clip.append(jittered_img.astype("float32"))

        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue \
                = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: adjust_brightness(img, brightness))
            if contrast is not None:
                img_transforms.append(lambda img: adjust_contrast(img, contrast))
            if saturation is not None:
                img_transforms.append(lambda img: adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: adjust_hue(img, hue))

            random.shuffle(img_transforms)

            jittered_clip = []
            for c in clip:
                for img_transform in img_transforms:
                    jittered_img = img_transform(c)
                    jittered_clip.append(jittered_img)

        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image," + f"but got a list of {type(clip[0])}")

        return jittered_clip


class AllAugmentationTransform:
    def __init__(self, flip_param=None, jitter_param=None):
        self.transforms = []

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))

    def __call__(self, clip):
        for transform in self.transforms:
            clip = transform(clip)
        return clip
