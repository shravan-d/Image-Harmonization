from models.AttUNet import UnetGenerator, UNet_3Plus
from models.BargaiNet import StyleEncoder
from loss.LossFunctions import StyleLoss
from models.util import *
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
learning_rate = 0.002
batch_size = 1
loss_style = StyleLoss()
loss_L1 = torch.nn.L1Loss()
loss_triplet = nn.TripletMarginLoss(margin=1, p=2)
content_loss_weight = 0.0
style_loss_weight = 0.01
disc_loss_weight = 0.01
epochs = 8
loss_functions = ['L1', 'style', 'discriminator']

from data import create_dataset
opt = {'dataset_mode': 'HCOCO', 'dataset_root': 'HCOCO/', 'serial_batches': True, 'batch_size': 1, 'num_threads': 2,
       'isTrain': True, 'preprocess': 'resize_and_crop', 'no_flip': True, 'max_dataset_size': float("inf"),
       'load_size': 286, 'crop_size': 256}
dataset = create_dataset(opt)


def UNet3Plus_train():
    Unet3Gen = UNet_3Plus().to(device)
    Unet3Gen = init_weights(Unet3Gen, 'xavier')
    optimizerG = torch.optim.Adam(Unet3Gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # , weight_decay=1e-4)
    StyleDisc = StyleEncoder(16).to(device)
    StyleDisc = init_weights(StyleDisc, 'xavier')
    optimizerD = torch.optim.Adam(StyleDisc.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # , weight_decay=1e-4)
    for epoch in range(epochs):
        epoch_loss = 0
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            inputs = torch.cat([composite_image, mask],1).to(device)
            disc_bg_image = StyleDisc(real_image, bg_mask)
            bg_sty_map = disc_bg_image.expand([1,16,256,256])
            inputs_c2r = torch.cat([inputs, bg_sty_map],1)
            output = Unet3Gen(inputs_c2r)
            loss = 0
            if 'L1' in loss_functions:
                loss = loss_L1(output, real_image)
            if 'style' in loss_functions:
                loss_sty = loss_style(output, real_image, mask)
                loss += style_loss_weight * loss_sty
                loss_sty.detach_()
            if 'discriminator' in loss_functions:
                disc_real_image = StyleDisc(real_image, mask)
                disc_harm_image = StyleDisc(output, mask)
                disc_comp_image = StyleDisc(composite_image, mask)
                loss_disc = ((loss_triplet(disc_real_image, disc_harm_image, disc_comp_image)*1.0)
                + (loss_triplet(disc_harm_image, disc_bg_image, disc_comp_image)*0.5))*disc_loss_weight
                loss += disc_loss_weight*loss_disc
                loss_disc.detach_()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            loss.backward()
            optimizerG.step()
            optimizerD.step()
            epoch_loss += loss.item()
        print('\nEpoch: '+str(epoch+1)+'; Model saved with loss: '+ '%.2f' % (epoch_loss / dataset_size))
    return Unet3Gen, StyleDisc


def ModelRGB_train():
    UnetGen = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    UnetGen = init_weights(UnetGen, 'xavier')
    optimizerG = torch.optim.Adam(UnetGen.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # , weight_decay=1e-4)
    StyleDisc = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
    StyleDisc = init_weights(StyleDisc, 'xavier')
    optimizerD = torch.optim.Adam(StyleDisc.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # , weight_decay=1e-4)
    # schedulerG = lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_rule)
    # schedulerD = lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_rule)
    for epoch in range(epochs):
        epoch_loss = 0
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            inputs = torch.cat([composite_image, mask], 1).to(device)
            disc_bg_image = StyleDisc(real_image, bg_mask)
            disc_real_image = StyleDisc(real_image, mask)
            disc_comp_image = StyleDisc(composite_image, mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = UnetGen(inputs_c2r)
            disc_harm_image = StyleDisc(output, mask)
            loss = 0
            if 'L1' in loss_functions:
                loss = loss_L1(output, real_image)
            if 'style' in loss_functions:
                loss_sty = loss_style(output, real_image, mask)
                loss += style_loss_weight * loss_sty
                loss_sty.detach_()
            if 'discriminator' in loss_functions:
                loss_disc = ((loss_triplet(disc_real_image, disc_harm_image, disc_comp_image) * 1.0)
                             + (loss_triplet(disc_harm_image, disc_bg_image, disc_comp_image) * 0.5)) * disc_loss_weight
                loss += loss_disc
                loss_disc.detach_()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            loss.backward()
            optimizerG.step()
            optimizerD.step()
            epoch_loss += loss.item()
        #         schedulerG.step()
        #         schedulerD.step()
        print('\nEpoch: ' + str(epoch + 1) + '; Model saved with loss: ' + '%.2f' % (epoch_loss / dataset_size))
    return UnetGen, StyleDisc


def ModelHSV_train():
    UnetGen = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
    UnetGen = init_weights(UnetGen, 'xavier')
    optimizerG = torch.optim.Adam(UnetGen.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # , weight_decay=1e-4)
    StyleDisc = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
    StyleDisc = init_weights(StyleDisc, 'xavier')
    optimizerD = torch.optim.Adam(StyleDisc.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # , weight_decay=1e-4)
    for epoch in range(epochs):
        epoch_loss = 0
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
            composite_image_hsv = dataset.dataset.input_transform(
                Image.fromarray(np.uint8(tensor2im(composite_image))).convert('HSV')).to(device)
            inputs = torch.cat([composite_image, mask], 1).to(device)
            disc_bg_image = StyleDisc(real_image_hsv.unsqueeze(0), bg_mask)
            disc_real_image = StyleDisc(real_image_hsv.unsqueeze(0), mask)
            disc_comp_image = StyleDisc(composite_image_hsv.unsqueeze(0), mask)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = UnetGen(inputs_c2r)
            output_hsv = dataset.dataset.input_transform(
                Image.fromarray(np.uint8(tensor2im(output))).convert('HSV')).to(device)
            disc_harm_image = StyleDisc(output_hsv.unsqueeze(0), mask)
            loss = 0
            if 'L1' in loss_functions:
                loss = loss_L1(output, real_image)
            if 'style' in loss_functions:
                loss_sty = loss_style(output, real_image, mask)
                loss += style_loss_weight * loss_sty
                loss_sty.detach_()
            if 'discriminator' in loss_functions:
                loss_disc = ((loss_triplet(disc_real_image, disc_harm_image, disc_comp_image) * 1.0)
                             + (loss_triplet(disc_harm_image, disc_bg_image, disc_comp_image) * 0.5)) * disc_loss_weight
                loss += loss_disc
                loss_disc.detach_()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            loss.backward()
            optimizerG.step()
            optimizerD.step()
            epoch_loss += loss.item()
        print('\nEpoch: ' + str(epoch + 1) + '; Model saved with loss: ' + '%.2f' % (epoch_loss / dataset_size))
    return UnetGen, StyleDisc


def ModelHLS_train():
    test_epoch = 8
    UnetGen, optimizerG, StyleDisc, optimizerD, epoch = get_model(learning_rate, device, False)
    while epoch < epochs:
        epoch_loss = 0
        dataset_size = len(dataset)
        start = datetime.datetime.now()
        for i, data in enumerate(dataset):
            progress_bar(start, i, dataset_size)
            composite_image = data['comp'].to(device)
            real_image = data['real'].to(device)
            mask = data['mask'].to(device)
            bg_mask = 1.0 - mask
            real_hls = rgb_to_hsl(Image.fromarray(np.uint8(tensor2im(real_image))))
            real_hls_tensor = dataset.dataset.input_transform(Image.fromarray(np.uint8(real_hls))).to(device)
            comp_hls = rgb_to_hsl(Image.fromarray(np.uint8(tensor2im(composite_image))))
            comp_hls_tensor = dataset.dataset.input_transform(Image.fromarray(np.uint8(comp_hls))).to(device)

            disc_bg_image = StyleDisc(comp_hls_tensor.unsqueeze(0), bg_mask)
            disc_real_image = StyleDisc(real_hls_tensor.unsqueeze(0), mask)
            disc_comp_image = StyleDisc(comp_hls_tensor.unsqueeze(0), mask)

            inputs = torch.cat([composite_image, mask], 1).to(device)
            bg_sty_map = disc_bg_image.expand([1, 16, 256, 256])
            inputs_c2r = torch.cat([inputs, bg_sty_map], 1)
            output = UnetGen(inputs_c2r)
            harm_hls = rgb_to_hsl(Image.fromarray(np.uint8(tensor2im(output))))
            harm_hls_tensor = dataset.dataset.input_transform(Image.fromarray(np.uint8(harm_hls))).to(device)
            disc_harm_image = StyleDisc(harm_hls_tensor.unsqueeze(0), mask)
            loss = 0
            if 'L1' in loss_functions:
                loss = loss_L1(output, real_image)
            if 'style' in loss_functions:
                loss_sty = loss_style(output, real_image, mask)
                loss += style_loss_weight * loss_sty
                loss_sty.detach_()
            if 'discriminator' in loss_functions:
                loss_disc = ((loss_triplet(disc_real_image, disc_harm_image, disc_comp_image) * 1.0)
                             + (loss_triplet(disc_harm_image, disc_bg_image, disc_comp_image) * 0.9)) * disc_loss_weight
                loss += loss_disc
                loss_disc.detach_()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            loss.backward()
            optimizerG.step()
            optimizerD.step()
            epoch_loss += loss.item()
        print('\nEpoch: ' + str(epoch + 1) + '; Model saved with loss: ' + '%.2f' % (epoch_loss / dataset_size))
        torch.save({
            'epoch': epoch,
            'model_state_dict': UnetGen.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict()
        }, 'models/HSLGen.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': StyleDisc.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict()
        }, 'models/HSLDisc.pt')
    return UnetGen, StyleDisc
