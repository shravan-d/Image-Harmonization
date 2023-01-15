import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import cv2
from albumentations import HorizontalFlip, RandomResizedCrop, Compose, DualTransform
import albumentations.augmentations.transforms as transforms


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt['dataset_root']

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


# Bargain Net
class HCompose(Compose):
    def __init__(self, transforms, *args, additional_targets=None, no_nearest_for_masks=True, **kwargs):
        if additional_targets is None:
            additional_targets = {
                'real': 'image',
                'mask': 'mask'
            }
        self.additional_targets = additional_targets
        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)
        if no_nearest_for_masks:
            for t in transforms:
                if isinstance(t, DualTransform):
                    t._additional_targets['mask'] = 'image'


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt['load_size']
    elif opt['preprocess'] == 'scale_width_and_crop':
        new_w = opt['load_size']
        new_h = opt['load_size'] * h // w

    x = random.randint(0, np.maximum(0, new_w - opt['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - opt['crop_size']))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


# DoveNet
# def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
#     transform_list = []
#     if grayscale:
#         transform_list.append(transforms.Grayscale(1))
#     if 'resize' in opt['preprocess']:
#         osize = [opt['load_size'], opt['load_size']]
#         transform_list.append(transforms.Resize(osize, method))
#     elif 'scale_width' in opt.preprocess:
#         transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt['load_size'], method)))

#     if 'crop' in opt['preprocess']:
#         if params is None:
#             transform_list.append(transforms.RandomCrop(opt['crop_size']))
#         else:
#             transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt['crop_size'])))

#     if opt['preprocess'] == 'none':
#         transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

#     if not opt['no_flip']:
#         if params is None:
#             transform_list.append(transforms.RandomHorizontalFlip())
#         elif params['flip']:
#             transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

#     if convert:
#         transform_list += [transforms.ToTensor()]
#         if grayscale:
#             transform_list += [transforms.Normalize((0.5,), (0.5,))]
#         else:
#             transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     return transforms.Compose(transform_list)

# Bargain Net
def get_transform(opt, params=None, grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.ToGray())
    if opt['preprocess'] == 'resize_and_crop':
        if params is None:
            transform_list.append(RandomResizedCrop(256, 256, scale=(0.5, 1.0)))
    elif opt['preprocess'] == 'resize':
        transform_list.append(transforms.Resize(256, 256))

    if not opt['no_flip']:
        if params is None:
            transform_list.append(HorizontalFlip())

    return HCompose(transform_list)


def __make_power_2(img, base):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
