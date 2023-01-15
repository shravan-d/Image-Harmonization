from models.AttUNet import UnetGenerator, UNet_3Plus
from models.BargaiNet import StyleEncoder
from models.util import *
import sys
import torch
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import torch.nn as nn
from data import create_dataset

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

opt = {'dataset_mode': 'HCOCO', 'dataset_root': 'HCOCO/', 'serial_batches': True, 'batch_size': 1, 'num_threads': 2,
           'isTrain': False, 'preprocess': 'resize_and_crop', 'no_flip': True, 'max_dataset_size': float("inf"),
           'load_size': 286, 'crop_size': 256}
dataset = create_dataset(opt)


def bargain_net_test():
    modelG2 = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    modelG2.load_state_dict(torch.load('models/latest_net_G.pth'))
    modelD2 = StyleEncoder(16).to(device)
    modelD2.load_state_dict(torch.load('models/latest_net_E.pth'))
    mse_loss = 0
    psnr_loss = 0
    with torch.no_grad():
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            inputs = torch.cat([composite_image, mask], 1).to(device)
            bg_mask = 1.0 - mask
            disc_real_image = modelD2(real_image, bg_mask)
            bg_sty_map = disc_real_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output_image = modelG2(inputs_c2r)
            output_image = np.array(tensor2im(output_image), dtype=np.float32)
            real_image = np.array(tensor2im(real_image), dtype=np.float32)
            #         mask = np.array(tensor2im(mask), dtype=np.float32)
            #         mask = 0 + (mask > 150)
            #         output_image = real_image*(1-mask) + output_image*mask
            mse_loss += mse(output_image, real_image)
            psnr_loss += psnr(output_image, real_image, data_range=output_image.max() - output_image.min())
        sys.stdout.write('\r\n')
        print("MSE: " + '%.2f' % (mse_loss / dataset_size))
        print("PSNR: " + '%.2f' % (psnr_loss / dataset_size))


def UNet3Plus_test(Unet3Gen=None, StyleDisc3=None):
    if Unet3Gen is None:
        Unet3Gen = UNet_3Plus().to(device)
        Unet3Gen.load_state_dict(torch.load('models/Unet3RGB.pth'))
    if StyleDisc3 is None:
        StyleDisc3 = StyleEncoder(16).to(device)
        StyleDisc3.load_state_dict(torch.load('models/StyleDisc3RGB.pth'))
    mse_loss = 0
    psnr_loss = 0
    with torch.no_grad():
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            inputs = torch.cat([composite_image, mask], 1).to(device)
            disc_bg_image = StyleDisc3(real_image, bg_mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = Unet3Gen(inputs_c2r)
            output_image = np.array(tensor2im(output), dtype=np.float32)
            real_image = np.array(tensor2im(real_image), dtype=np.float32)
            mask = np.array(tensor2im(mask), dtype=np.float32)
            mask = 0 + (mask > 150)
            output_image = real_image * (1 - mask) + output_image * mask
            mse_loss += mse(output_image, real_image)
            psnr_loss += psnr(output_image, real_image, data_range=output_image.max() - output_image.min())
        sys.stdout.write('\r\n')
        print("MSE: " + '%.2f' % (mse_loss / dataset_size))
        print("PSNR: " + '%.2f' % (psnr_loss / dataset_size))


def ModelRGB_test(UnetGenRGB=None, StyleDiscRGB=None):
    if UnetGenRGB is None:
        UnetGenRGB = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        UnetGenRGB.load_state_dict(torch.load('models/UnetRGB.pth'))
    if StyleDiscRGB is None:
        StyleDiscRGB = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
        StyleDiscRGB.load_state_dict(torch.load('models/StyleDiscRGB.pth'))
    mse_loss = 0
    psnr_loss = 0
    with torch.no_grad():
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            inputs = torch.cat([composite_image, mask], 1).to(device)
            disc_bg_image = StyleDiscRGB(real_image, bg_mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = UnetGenRGB(inputs_c2r)
            output_image = np.array(tensor2im(output), dtype=np.float32)
            real_image = np.array(tensor2im(real_image), dtype=np.float32)
            #         mask = np.array(tensor2im(mask), dtype=np.float32)
            #         mask = 0 + (mask > 150)
            #         output_image = real_image*(1-mask) + output_image*mask
            mse_loss += mse(output_image, real_image)
            psnr_loss += psnr(output_image, real_image, data_range=output_image.max() - output_image.min())
        sys.stdout.write('\r\n')
        print("MSE: " + '%.2f' % (mse_loss / dataset_size))
        print("PSNR: " + '%.2f' % (psnr_loss / dataset_size))


def ModelHSV_test(UnetGenHSV=None, StyleDiscHSV=None):
    if UnetGenHSV is None:
        UnetGenHSV = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        UnetGenHSV.load_state_dict(torch.load('models/UnetHSV.pth'))
    if StyleDiscHSV is None:
        StyleDiscHSV = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
        StyleDiscHSV.load_state_dict(torch.load('models/StyleDiscHSV.pth'))
    mse_loss = 0
    psnr_loss = 0
    with torch.no_grad():
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            real_image_hsv = dataset.dataset.input_transform(
                Image.fromarray(np.uint8(tensor2im(real_image))).convert('HSV')).to(device)
            inputs = torch.cat([composite_image, mask], 1).to(device)
            disc_bg_image = StyleDiscHSV(real_image_hsv.unsqueeze(0), bg_mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = UnetGenHSV(inputs_c2r)
            output_image = np.array(tensor2im(output), dtype=np.float32)
            real_image = np.array(tensor2im(real_image), dtype=np.float32)
            #         mask = np.array(tensor2im(mask), dtype=np.float32)
            #         mask = 0 + (mask > 150)
            #         output_image = real_image*(1-mask) + output_image*mask
            mse_loss += mse(output_image, real_image)
            psnr_loss += psnr(output_image, real_image, data_range=output_image.max() - output_image.min())
        sys.stdout.write('\r\n')
        print("MSE: " + '%.2f' % (mse_loss / dataset_size))
        print("PSNR: " + '%.2f' % (psnr_loss / dataset_size))


def ModelHSL_test(UnetGen=None, StyleDisc=None):
    if UnetGen is None:
        UnetGen = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        UnetGen.load_state_dict(torch.load('models/UnetHSV.pth'))
    if StyleDisc is None:
        StyleDisc = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
        StyleDisc.load_state_dict(torch.load('models/StyleDiscHSV.pth'))
    mse_loss = 0
    psnr_loss = 0
    with torch.no_grad():
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            comp_hls = rgb_to_hsl(Image.fromarray(np.uint8(tensor2im(composite_image))))
            comp_hls_tensor = dataset.dataset.input_transform(Image.fromarray(np.uint8(comp_hls))).to(device)

            inputs = torch.cat([composite_image, mask], 1).to(device)
            disc_bg_image = StyleDisc(comp_hls_tensor.unsqueeze(0), bg_mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = UnetGen(inputs_c2r)
            output_image = np.array(tensor2im(output), dtype=np.float32)
            real_image = np.array(tensor2im(real_image), dtype=np.float32)
            mse_loss += mse(output_image, real_image)
            psnr_loss += psnr(output_image, real_image, data_range=output_image.max() - output_image.min())
        sys.stdout.write('\r\n')
        print("MSE: " + '%.2f' % (mse_loss / dataset_size))
        print("PSNR: " + '%.2f' % (psnr_loss / dataset_size))


def Ensemble_test(UnetGenRGB=None, StyleDiscRGB=None, modelG2=None, modelD2=None):
    if modelG2 is None:
        modelG2 = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        modelG2.load_state_dict(torch.load('models/latest_net_G.pth'))
    if modelD2 is None:
        modelD2 = StyleEncoder(16).to(device)
        modelD2.load_state_dict(torch.load('models/latest_net_E.pth'))
    if UnetGenRGB is None:
        UnetGenRGB = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        UnetGenRGB.load_state_dict(torch.load('models/UnetRGB.pth'))
    if StyleDiscRGB is None:
        StyleDiscRGB = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
        StyleDiscRGB.load_state_dict(torch.load('models/StyleDiscRGB.pth'))
    mse_loss = 0
    psnr_loss = 0
    with torch.no_grad():
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            inputs = torch.cat([composite_image, mask], 1).to(device)

            # Model 1
            disc_real_image = modelD2(real_image, bg_mask)
            bg_sty_map = disc_real_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output_b1 = modelG2(inputs_c2r)

            # Model 2
            disc_bg_image = StyleDiscRGB(real_image, bg_mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output_b2 = UnetGenRGB(inputs_c2r)

            output = torch.mean(torch.stack([output_b1, output_b2]), dim=0)

            output_image = np.array(tensor2im(output), dtype=np.float32)
            real_image = np.array(tensor2im(real_image), dtype=np.float32)
            mse_loss += mse(output_image, real_image)
            psnr_loss += psnr(output_image, real_image, data_range=output_image.max() - output_image.min())

        print("MSE: " + '%.2f' % (mse_loss / dataset_size))
        print("PSNR: " + '%.2f' % (psnr_loss / dataset_size))
