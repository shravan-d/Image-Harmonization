import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
import cv2
import numpy as np
import torchvision.transforms as transforms
import random


class HCOCODataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"),
                            new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt['isTrain']
        if opt['isTrain']:
            print('loading training file: ')
            self.keep_background_prob = 0.05
            self.trainfile = opt['dataset_root'] + 'HCOCO_train.txt'
            with open(self.trainfile, 'r') as f:
                for line in f.readlines():
                    self.image_paths.append(os.path.join(opt['dataset_root'] + 'composite_images', line.rstrip()))
        elif not opt['isTrain']:
            print('loading test file')
            self.keep_background_prob = -1
            self.trainfile = opt['dataset_root'] + 'HCOCO_test.txt'
            with open(self.trainfile, 'r') as f:
                for line in f.readlines():
                    self.image_paths.append(os.path.join(opt['dataset_root'] + 'composite_images', line.rstrip()))
        self.transform = get_transform(opt)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # DoveNet
    #     def __getitem__(self, index):

    #         path = self.image_paths[index]
    #         name_parts = path.split('_')
    #         mask_path = self.image_paths[index].replace('composite_images', 'masks')
    #         mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')
    #         target_path = self.image_paths[index].replace('composite_images', 'real_images')
    #         target_path = target_path.replace(('_' + name_parts[-2] + '_' + name_parts[-1]), '.jpg')

    #         comp = Image.open(path).convert('RGB')
    #         real = Image.open(target_path).convert('RGB')
    #         mask = Image.open(mask_path).convert('1')

    #         comp = tf.resize(comp, [256, 256])
    #         mask = tf.resize(mask, [256, 256])
    #         real = tf.resize(real, [256, 256])

    #         comp = self.transform(comp)
    #         mask = tf.to_tensor(mask)
    #         real = self.transform(real)
    #         inputs = torch.cat([comp, mask], 0)

    #         return {'inputs': inputs, 'comp': comp, 'real': real, 'img_path': path}

    # Bargain Net
    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)
        comp = self.input_transform(sample['image'])
        real = self.input_transform(sample['real'])
        mask = sample['mask'].astype(np.float32)
        output = {
            'comp': comp,
            'mask': mask[np.newaxis, ...].astype(np.float32),
            'real': real,
            'img_path': sample['img_path']
        }
        return output

    def check_sample_types(self, sample):
        assert sample['comp'].dtype == 'uint8'
        if 'real' in sample:
            assert sample['real'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.transform is None:
            return sample
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.transform(image=sample['comp'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['mask'].sum() > 1.0

    def get_sample(self, index):
        path = self.image_paths[index]
        name_parts = path.split('_')
        mask_path = self.image_paths[index].replace('composite_images', 'masks')
        mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')
        target_path = self.image_paths[index].replace('composite_images', 'real_images')
        target_path = target_path.replace(('_' + name_parts[-2] + '_' + name_parts[-1]), '.jpg')

        comp = cv2.imread(path)
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        real = cv2.imread(target_path)
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32) / 255.
        mask = mask.astype(np.uint8)

        return {'comp': comp, 'mask': mask, 'real': real, 'img_path': path}

    def __len__(self):
        return len(self.image_paths)