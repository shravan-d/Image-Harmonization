from torch.nn import init
import numpy as np
import torch
import datetime
import sys
import colorsys
from models.AttUNet import UnetGenerator, UNet_3Plus
from models.BargaiNet import StyleEncoder
import torch.nn as nn


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


def progress_bar(start, i, training_batch_count):
    elapsed_time = (datetime.datetime.now() - start).seconds // 60
    sys.stdout.write('\r')
    sys.stdout.write("Training: [%-50s] %d%% || ETA: %d minutes"
                     % ('=' * int(50 * (i + 1) / training_batch_count), int(100 * (i + 1) / training_batch_count),
                        (elapsed_time / (i+1)) * (training_batch_count - i)))
    sys.stdout.flush()


def rgb_to_hsl(img):
    vals = list(img.getdata())
    l = len(vals)
    side = 256
    c1 = np.zeros(shape=(side, side), dtype=float)
    c2 = np.zeros(shape=(side, side), dtype=float)
    c3 = np.zeros(shape=(side, side), dtype=float)

    k = 0
    for i in range(side):
      for j in range(side):
          if k >= l:
              break

          r = float(vals[k][0]) / 255
          g = float(vals[k][1]) / 255
          b = float(vals[k][2]) / 255
          c1[i, j], c2[i, j], c3[i, j] = colorsys.rgb_to_hls(r, g, b)
          k += 1
    real_hls = np.stack([c1, c2, c3])
    real_hls = np.transpose(real_hls, (1, 2, 0))
    return real_hls


def get_model(learning_rate, device, load_chk=False):
    if load_chk:
        UnetGen = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        optimizerG = torch.optim.Adam(UnetGen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        checkpoint = torch.load('models/HSLGen.pt')
        UnetGen.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        StyleDisc = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
        optimizerD = torch.optim.Adam(StyleDisc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        checkpoint = torch.load('models/HSLDisc.pt')
        StyleDisc.load_state_dict(checkpoint['model_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        UnetGen = UnetGenerator(20, 3, 8, 64, nn.BatchNorm2d, False, use_attention=True).to(device)
        UnetGen = init_weights(UnetGen, 'xavier')
        optimizerG = torch.optim.Adam(UnetGen.parameters(), lr=learning_rate,
                                      betas=(0.5, 0.999))  # , weight_decay=1e-4)
        StyleDisc = StyleEncoder(16, norm_layer=nn.InstanceNorm2d).to(device)
        StyleDisc = init_weights(StyleDisc, 'xavier')
        optimizerD = torch.optim.Adam(StyleDisc.parameters(), lr=learning_rate,
                                      betas=(0.5, 0.999))  # , weight_decay=1e-4)
        epoch = 0
    return UnetGen, optimizerG, StyleDisc, optimizerD, epoch
