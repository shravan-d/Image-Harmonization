from models.AttUNet import UnetGenerator, UNet_3Plus
from models.BargaiNet import StyleEncoder
from models.util import *
import torch
from data import create_dataset
from PIL import Image
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

opt = {'dataset_mode': 'HCOCO', 'dataset_root': 'HCOCO/', 'serial_batches': True, 'batch_size': 1, 'num_threads': 2,
       'isTrain': False, 'preprocess': 'resize_and_crop', 'no_flip': True, 'max_dataset_size': float("inf"),
       'load_size': 286, 'crop_size': 256}
dataset = create_dataset(opt)


def predictEnsemble(real, comp, mask):
    modelG2 = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    modelG2.load_state_dict(torch.load('models/latest_net_G.pth'))
    modelD2 = StyleEncoder(16).to(device)
    modelD2.load_state_dict(torch.load('models/latest_net_E.pth'))
    UnetGenRGB = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    UnetGenRGB.load_state_dict(torch.load('models/UnetRGB.pth'))
    StyleDiscRGB = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
    StyleDiscRGB.load_state_dict(torch.load('models/StyleDiscRGB.pth'))

    bg_mask = 1.0 - mask
    inputs = torch.cat([comp, mask], 1).to(device)

    disc_real_image = modelD2(real, bg_mask)
    bg_sty_map = disc_real_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    output_b1 = modelG2(inputs_c2r)

    disc_bg_image = StyleDiscRGB(real, bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    output_b2 = UnetGenRGB(inputs_c2r)

    output = torch.mean(torch.stack([output_b1, output_b2]), dim=0)

    return output


def predict_all_util(comp, real, mask):
    Unet3Gen = UNet_3Plus().to(device)
    Unet3Gen.load_state_dict(torch.load('models/Unet3RGB.pth'))
    StyleDisc3 = StyleEncoder(16).to(device)
    StyleDisc3.load_state_dict(torch.load('models/StyleDisc3RGB.pth'))
    UnetGenRGB = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    UnetGenRGB.load_state_dict(torch.load('models/UnetRGB.pth'))
    StyleDiscRGB = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
    StyleDiscRGB.load_state_dict(torch.load('models/StyleDiscRGB.pth'))
    UnetGenHSV = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    UnetGenHSV.load_state_dict(torch.load('models/UnetHSV.pth'))
    StyleDiscHSV = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
    StyleDiscHSV.load_state_dict(torch.load('models/StyleDiscHSV.pth'))

    bg_mask = 1.0 - mask
    inputs = torch.cat([comp, mask], 1).to(device)
    disc_bg_image = StyleDisc3(real, bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    output3 = Unet3Gen(inputs_c2r)

    disc_bg_image = StyleDiscRGB(real, bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    outputRGB = UnetGenRGB(inputs_c2r)

    real_image_hsv = dataset.dataset.input_transform(Image.fromarray(np.uint8(tensor2im(real))).convert('HSV')).to(
        device)
    disc_bg_image = StyleDiscHSV(real_image_hsv.unsqueeze(0), bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    outputHSV = UnetGenHSV(inputs_c2r)

    return output3, outputRGB, outputHSV


def predict_all():
    idx = [37]  # 37 44
    for i, data in enumerate(dataset):
        comp = data['comp'].to(device)
        real = data['real'].to(device)
        mask = data['mask'].to(device)
        harm3, harmRGB, harmHSV = predict_all_util(comp, real, mask)
        harmEnsemble = predictEnsemble(real, comp, mask)
        comp = np.array(tensor2im(comp), dtype=np.float32)
        real = np.array(tensor2im(real), dtype=np.float32)
        mask = np.array(tensor2im(mask), dtype=np.float32)
        harm3 = np.array(tensor2im(harm3), dtype=np.float32)
        harmRGB = np.array(tensor2im(harmRGB), dtype=np.float32)
        harmHSV = np.array(tensor2im(harmHSV), dtype=np.float32)
        harmEnsemble = np.array(tensor2im(harmEnsemble), dtype=np.float32)
        mask_disp = np.where(mask > 127, 255, 0)
        mask = np.where(mask > 127, 1, 0)
        harm3 = real * (1 - mask) + harm3 * mask
        harmRGB = real * (1 - mask) + harmRGB * mask
        harmHSV = real * (1 - mask) + harmHSV * mask
        if i in idx:
            fig, axarr = plt.subplots(1, 3, figsize=(15, 8))
            imgs = [comp, real, mask_disp]
            imgs_tit = ['composite', 'real', 'mask']
            for i, (ax, im) in enumerate(zip(axarr.ravel(), imgs)):
                ax.imshow(im.astype(np.uint8))
                ax.title.set_text(imgs_tit[i])
            fig2, axarr2 = plt.subplots(2, 2, figsize=(12, 12))
            imgs = [harm3, harmRGB, harmHSV, harmEnsemble]
            imgs_tit = ['Unet3Plus', 'VGG(RGB)', 'VGG(HSV)', 'Ensemble']
            for i, (ax, im) in enumerate(zip(axarr2.ravel(), imgs)):
                ax.imshow(im.astype(np.uint8))
                ax.title.set_text(imgs_tit[i])


def predict(comp, real, mask, modelG, modelD, is_hsv=False):
    bg_mask = 1.0 - mask
    inputs = torch.cat([comp, mask], 1).to(device)
    disc_bg_image = modelD(real, bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    output3 = modelG(inputs_c2r)

    disc_bg_image = modelD(real, bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    outputRGB = modelG(inputs_c2r)

    if is_hsv:
        real = dataset.dataset.input_transform(Image.fromarray(np.uint8(tensor2im(real))).convert('HSV')).to(
        device)
    disc_bg_image = modelD(real.unsqueeze(0), bg_mask)
    bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
    inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
    outputHSV = modelG(inputs_c2r)

    return output3, outputRGB, outputHSV
