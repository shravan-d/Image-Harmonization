from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class DataGenerator:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.composite_images = []
        self.real_images = []
        self.filenames_train = []
        self.filenames_test = []
        self.real_dir = images_dir + 'real_images/'
        self.composite_dir = images_dir + 'composite_images/'
        self.masks_dir = images_dir + 'masks/'
        self.transform = None
        self.transform_mask = None
        self.get_transforms()
        self.get_transforms(mask=True)
        self.get_filenames()

    def get_filenames(self):
        with open(self.images_dir + 'HCOCO_train.txt') as file:
            self.filenames_train = [line.rstrip() for line in file]
        with open(self.images_dir + 'HCOCO_test.txt') as file:
            self.filenames_test = [line.rstrip() for line in file]

    def get_filecount(self, mode):
        if mode == 'train':
            return len(self.filenames_train)
        else:
            return len(self.filenames_test)

    def get_transforms(self, input_dim=256, mask=False):
        transform_list = [transforms.Resize(286), transforms.CenterCrop(input_dim), transforms.ToTensor()]
        if mask:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
            self.transform_mask = transforms.Compose(transform_list)
        if not mask:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            self.transform = transforms.Compose(transform_list)

    def generate(self, mode, need_hsv=False, batch_size=32):
        current = 0
        if mode == 'train':
            filenames = self.filenames_train
        else:
            filenames = self.filenames_test
        while True:
            batch_composite, batch_real, batch_mask, batch_real_hsv, batch_composite_hsv = [], [], [], [], []
            batch_filenames = filenames[current:current + batch_size]
            current += batch_size
            for filename in batch_filenames:
                filename_real = filename.split('_')[0] + '.jpg'
                filename_mask = filename.split('_')[0] + '_' + filename.split('_')[1] + '.png'
                image1 = Image.open(self.composite_dir + filename)
                image2 = Image.open(self.real_dir + filename_real)
                image3 = Image.open(self.masks_dir + filename_mask).convert('L')
                batch_composite.append(image1)
                batch_real.append(image2)
                batch_mask.append(image3)
                if need_hsv:
                    batch_real_hsv.append(image2.convert('HSV'))
                    batch_composite_hsv.append(image1.convert('HSV'))
            for i in range(batch_size):
                batch_composite[i] = self.transform(batch_composite[i])
                batch_real[i] = self.transform(batch_real[i])
                batch_mask[i] = self.transform_mask(batch_mask[i])
                if need_hsv:
                    batch_composite_hsv[i] = self.transform(batch_composite_hsv[i])
                    batch_real_hsv[i] = self.transform(batch_real_hsv[i])
            if need_hsv:
                yield batch_composite, batch_real, batch_mask, batch_composite_hsv, batch_real_hsv
            else:
                yield batch_composite, batch_real, batch_mask
