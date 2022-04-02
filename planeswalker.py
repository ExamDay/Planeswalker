import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import linregress
from kornia import color
from time import time
from torchsummary import summary

import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
plt.rcParams["axes.facecolor"] = "black"

# CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
# CuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# DATA_CHUNKS = os.listdir(DATA_ROOT)

# Utils:
def display_images(in_, out=None, cols=4, n=1, label=None):
    for N in range(n):
        # in_pic = in_.squeeze(1).data.cpu().numpy()
        in_pic = in_.data.cpu().numpy()
        # in_pic = in_.data.cpu().view(-1, in_.shape[-1], in_.shape[-1])
        nPics = in_.shape[0]
        L = int(in_.shape[-1] / 128)
        f = plt.figure(figsize=(nPics * L / n, L + L * (out is not None)))
        for i in range(cols):
            plt.subplot(1 + (out is not None), cols, i + 1)
            # plt.imshow(in_pic[i+4*N])
            # plt.imshow(in_pic[i+4*N], cmap="gray")
            plt.imshow(np.transpose(in_pic[i + cols * N], (1, 2, 0)))
            plt.axis("off")
        if out is not None:
            # out_pic = out.squeeze(1).data.cpu().numpy()
            out_pic = out.data.cpu().numpy()
            # out_pic = out.data.cpu().view(-1, out.shape[-1], out.shape[-1])
            for i in range(cols):
                plt.subplot(2, cols, in_.shape[0] + i + 1)
                # plt.imshow(out_pic[i+4*N])
                # plt.imshow(out_pic[i+4*N], cmap="gray")
                plt.imshow(np.transpose(out_pic[i + cols * N], (1, 2, 0)))
                plt.axis("off")
        f.set_facecolor((0, 0, 0, 0.9))
        if label:
            plt.suptitle(label, color="white", fontsize=24)


def cas(dim, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """ calculate output shape of convolutional module """
    if type(dim) is int:
        dim = (dim, dim)
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    if type(dilation) is int:
        dilation = (dilation, dilation)
    if type(stride) is int:
        stride = (stride, stride)
    if type(padding) is int:
        padding = (padding, padding)

    def shape_each_dim(i):
        odim_i = dim[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1
        return int((odim_i / stride[i]) + 1)

    x = shape_each_dim(0)
    y = shape_each_dim(1)
    return x, y


def casT(dim, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """ calculate output shape of transpose convolutional module """
    if type(dim) is int:
        dim = (dim, dim)
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    if type(dilation) is int:
        dilation = (dilation, dilation)
    if type(stride) is int:
        stride = (stride, stride)
    if type(padding) is int:
        padding = (padding, padding)

    def shape_each_dim(i):
        odim_i = (dim[i] - 1) * stride[i]
        x = odim_i - 2 * padding[i] + dilation[i] * (kernel_size[i] - 1) + 1
        return x

    x = shape_each_dim(0)
    y = shape_each_dim(1)
    return x, y


def rgb_to_lab_batch(batch):
    # return batch  # <- easiest way to switch off and on.
    lab_batch = color.rgb_to_lab(batch)
    scaled_lab_batch = torch.zeros_like(lab_batch)
    for idx, pic in enumerate(lab_batch):
        scaled_lab_batch[idx][0] = torch.div(pic[0], 100)
        scaled_lab_batch[idx][1] = torch.div(torch.add(pic[1], 127), 127 * 2)
        scaled_lab_batch[idx][2] = torch.div(torch.add(pic[2], 127), 127 * 2)
    # display_images(scaled_lab_batch, cols=batch.shape[0], label="rgb to lab - lab")
    return scaled_lab_batch


def recursive_unpack(seq):
    """
        flattens nested sequences like [a, (b, c, d, (e, f)), g, [h, i, j], k]
        into lists like [a, b, c, d, e, f, g, h, i, j, k]
    """
    if seq in ([], ()):
        return seq
    if isinstance(seq[0], (list, tuple)):
        return list(recursive_unpack(seq[0])) + list(recursive_unpack(seq[1:]))
    return list(seq[:1]) + list(recursive_unpack(seq[1:]))


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        x = self.interp(x, *self.args, **self.kwargs)
        return x


class InitNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InitNorm, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.first = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.target_mean = 0
        self.target_std = 1
        self.constrain_mean = True
        try:
            self.target_mean = args[0]
            self.target_std = args[1]
        except IndexError:
            pass
        try:
            self.constrain_mean = kwargs["constrain_mean"]
        except KeyError:
            pass

    def forward(self, x):
        shape = x.shape
        x = x.flatten(1)
        if self.first == 1:
            std, mean = torch.std_mean(x, 0)
            self.mean = nn.Parameter(mean, requires_grad=False)
            self.std = nn.Parameter(std, requires_grad=False)
            self.first.data = torch.Tensor([0])
        if self.constrain_mean:
            return (
                ((x - self.mean) / self.std) * self.target_std + self.target_mean
            ).reshape(shape)
        else:
            return (
                (((x - self.mean) / self.std) * self.target_std + self.target_mean)
                + self.mean
            ).reshape(shape)


class ContainNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContainNorm, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.first = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.responsiveness = 0.95
        assert self.responsiveness <= 1 and self.responsiveness > 0

    def exp_moving_average(self, x, prev_x):
        return (x * self.responsiveness) + (prev_x * (1 - self.responsiveness))

    def forward(self, x):
        shape = x.shape
        x = x.flatten(1)
        if self.train:
            if self.first == 1:
                print("inializing containment")
                std, mean = torch.std_mean(x, 0)
                self.ema_mean = nn.Parameter(mean, requires_grad=False)
                self.ema_std = nn.Parameter(std, requires_grad=False)
                self.first.data = torch.Tensor([0])
            else:
                std, mean = torch.std_mean(x, 0)
                ema_mean = self.exp_moving_average(mean, self.ema_mean)
                ema_std = self.exp_moving_average(std, self.ema_std)
                self.ema_mean.data = ema_mean
                self.ema_std.data = ema_std
        return ((x - self.ema_mean) / self.ema_std).reshape(shape)


class Conv2d_With_Kernel_Injection(nn.Module):
    def __init__(
        self,
        num_injection_kernels,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding="same",
        dilation: int = 1,
    ):
        super(Conv2d_With_Kernel_Injection, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_injection_kernels = num_injection_kernels
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.d = dilation

        if out_channels - num_injection_kernels > 0:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels - num_injection_kernels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

    def inject(self, x, kernelBatch):
        # kernelBatch ~ [32, 5, 3, 3] (if kernelBatch is 32 instances of 5 channel 3x3 kernels)
        batches = kernelBatch.shape[0]
        nk = kernelBatch.shape[1]
        kw = kernelBatch.shape[2]
        kh = kernelBatch.shape[3]
        # stack up all the batches:
        kernels = kernelBatch.reshape(batches * nk, kw, kh)
        # kernels ~ [160, 3, 3]
        # shape into convolution weights:
        kernels = kernels.unsqueeze(1)
        # kernels ~ [160, 1, 3, 3]
        weights = kernels.repeat(1, self.in_channels, 1, 1)
        # weights ~ [160, 16, 3, 3] (if self.in_channels = 16)
        assert x.shape[1] == self.in_channels
        output = F.conv2d(
            x.view(1, batches * self.in_channels, x.shape[-2], x.shape[-1]),
            # x ~ [1, 512, 128, 128] if x was 32 instances of 16 channel 128x128 images.
            weights,
            stride=self.s,
            padding=self.p,
            dilation=self.d,
            groups=batches,  # ensures beams don't cross (batches don't interfere with eachother)
        )
        # output = [1, 160, 128, 128]
        return output.view(batches, nk, output.shape[-2], output.shape[-1])

    def forward(self, x, kernelBatch):
        injection = self.inject(x, kernelBatch)
        if self.out_channels - self.num_injection_kernels > 0:
            y = self.conv(x)
            out = torch.cat((injection, y), 1)
            return out
        else:
            return injection


class Injection_Block(nn.Module):
    def __init__(
        self,
        num_injection_kernels,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding="same",
        oDims=None,
        activation=nn.ReLU,
    ):
        super(Injection_Block, self).__init__()

        self.cki = Conv2d_With_Kernel_Injection(
            num_injection_kernels,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # self.normalizer = nn.InstanceNorm2d(out_channels, affine=True)
        # self.normalizer = nn.LayerNorm([out_channels, oDims[0], oDims[1]], elementwise_affine=True)
        # self.normalizer = nn.BatchNorm2d(out_channels)

        self.normalizerA = nn.LayerNorm(
            [out_channels, oDims[0], oDims[1]], elementwise_affine=True
        )

        # self.normalizerB = InitNorm(constrain_mean = False)
        # self.normalizerB = nn.BatchNorm2d(out_channels, affine=False)
        # self.normalizerB = ContainNorm()

        # self.normalizerC = nn.BatchNorm2d(out_channels, affine=False)
        # self.normalizerC = nn.LayerNorm([out_channels, oDims[0], oDims[1]], elementwise_affine=False)

        # self.regulizer = nn.Dropout(0.1)

        self.activation = nn.Sequential(
            # activation(),
            nn.LeakyReLU(0.2)
        )

        # self.contain = ContainNorm()

    def forward(self, x, kernelBatch):
        y = self.cki(x, kernelBatch)
        # y = self.regulizer(y)
        y = self.activation(y)
        y = self.normalizerA(y)
        # y = self.normalizerB(y)
        # y = self.normalizerC(y)
        # y = self.contain(y)
        return y


# class Decoder(nn.Module):
#     def __init__(self, z_dim, channels_img, features_gen=16, activation=nn.ReLU):
#         super(Decoder, self).__init__()
#         self.activation = activation
# dae = torch.load("models/DAE_dae_128_1_animefaces.ckpt").to(device)

#         k = 7
#         s = 1
#         p = 3

#         num_zk = 8
#         self.num_zk = num_zk

#         self.k = k
#         self.s = s
#         self.p = p

#         dims0 = (16, 16)
#         # dims1 = casT(dims0, kernel_size=k, stride=s, padding=p)
#         # dims2 = casT(dims1, kernel_size=k, stride=s, padding=p)
#         # dims3 = casT(dims2, kernel_size=k, stride=s, padding=p)
#         # dims4 = casT(dims3, kernel_size=k, stride=s, padding=p)
#         # dims5 = casT(dims4, kernel_size=k, stride=s, padding=p)

#         self.injection_blockA = Injection_Block(num_zk, 1, num_zk, k, s, p, (2, 2))
#         self.injection_blockA2 = Injection_Block(num_zk, num_zk, features_gen, k, s, p, (2, 2))

#         self.lA = nn.Sequential(
#             # self._block(num_zk, features_gen, k, s, p, (2, 2)),
#             # self._block(features_gen, features_gen, k, s, p, (2, 2)),
#             nn.Upsample((4, 4), mode="bilinear"), # 2x2 => 4x4
#         )


#         self.injection_blockB = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (4, 4))
#         self.injection_blockB2 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (4, 4))

#         self.lB = nn.Sequential(
#             # self._block(features_gen, features_gen, k, s, p, (4, 4)),
#             # self._block(features_gen, features_gen, k, s, p, (4, 4)),
#             nn.Upsample((8, 8), mode="bilinear"), # 4x4 => 8x8
#         )

#         self.injection_blockC = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (8, 8))
#         self.injection_blockC2 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (8, 8))

#         self.lC = nn.Sequential(
#             # self._block(features_gen, features_gen, k, s, p, (8, 8)),
#             # self._block(features_gen, features_gen, k, s, p, (8, 8)),
#             nn.Upsample((16, 16), mode="bilinear"), # 8x8 => 16x16
#         )

#         self.injection_block0 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (16, 16))
#         self.injection_block02 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (16, 16))

#         self.l0 = nn.Sequential(
#             # self._block(features_gen, features_gen, k, s, p, (16, 16)),
#             # self._block(features_gen, features_gen, k, s, p, (16, 16)),
#             nn.Upsample((32, 32), mode="bilinear"), # 16x16 => 32x32
#         )

#         #############################
#             # SAVE
#         #############################
#         # self.l0 = nn.Sequential(
#         #     self._block(4, features_gen, k, s, p, (16, 16)),
#         #     self._block(features_gen, features_gen, k, s, p, (16, 16)),
#         #     # self._block(features_gen, features_gen, k, s, p, (16, 16)),
#         #     nn.Upsample((32, 32), mode="bilinear"), # 16x16 => 32x32
#         # )
#         #############################

#         self.injection_block1 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (32, 32))
#         self.injection_block12 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (32, 32))

#         self.l1 = nn.Sequential(
#             # self._block(features_gen, features_gen, k, s, p, (32, 32)),
#             # self._block(features_gen, features_gen, k, s, p, (32, 32)),
#             nn.Upsample((64, 64), mode="bilinear"), # 32x32 => 64x64
#         )

#         self.injection_block2 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (64, 64))
#         self.injection_block22 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (64, 64))

#         self.l2 = nn.Sequential(
#             # self._block(features_gen, features_gen, k, s, p, (64, 64)),
#             # self._block(features_gen, features_gen, k, s, p, (64, 64)),
#             nn.Upsample((128, 128), mode="bilinear"), # 64x64 => 128x128
#         )

#         self.injection_block3 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (128, 128))
#         self.injection_block32 = Injection_Block(num_zk, features_gen, features_gen, k, s, p, (128, 128))

#         self.l3 = nn.Sequential(
#             # self._block(features_gen, features_gen, k, s, p, (128, 128)),
#             # self._block(features_gen, features_gen, k, s, p, (128, 128)),
#             # nn.Upsample((256, 256), mode="bilinear"), # 128x128 => 256x256

#             # self._block(features_gen, features_gen, k, s, p, (128, 128)),
#             # self._block(features_gen, features_gen, k, s, p, (128, 128)),
#             # nn.Conv2d(
#             #     features_gen,
#             #     features_gen,
#             #     kernel_size=k,
#             #     stride=s,
#             #     padding=p,
#             # ),
#             # nn.LeakyReLU(0.2),
#             nn.Conv2d(
#                 features_gen,
#                 channels_img,
#                 kernel_size=k,
#                 stride=s,
#                 padding=p,
#             ),
#             InitNorm(0, 0.25),
#             nn.Sigmoid(),
#             ## Output: N x channels_img x 128 x 128
#         )

#         # self.injection_block4 = Injection_Block(features_gen, features_gen, k, s, p, (256, 256))

#         # self.l4 = nn.Sequential(
#         #     self._block(features_gen, features_gen, k, s, p, (256, 256)),
#         #     self._block(features_gen, features_gen, k, s, p, (256, 256)),
#         #     nn.Upsample((512, 512), mode="bilinear"), # 256x256 => 512x512

#         #     # self._block(features_gen, features_gen, k, s, p, (256, 256)),
#         #     # self._block(features_gen, features_gen, k, s, p, (256, 256)),
#         #     # nn.Conv2d(
#         #     #     features_gen,
#         #     #     channels_img,
#         #     #     kernel_size=k,
#         #     #     stride=s,
#         #     #     padding=p,
#         #     # ),
#         #     # nn.Sigmoid(),
#         #     ## Output: N x channels_img x 256 x 256
#         # )

#         # self.injection_block5 = Injection_Block(features_gen, features_gen, k, s, p, (512, 512))

#         # self.l5 = nn.Sequential(
#         #     self._block(features_gen, features_gen, k, s, p, (512, 512)),
#         #     self._block(features_gen, features_gen, k, s, p, (512, 512)),
#         #     nn.Conv2d(
#         #         features_gen,
#         #         channels_img,
#         #         kernel_size=k,
#         #         stride=s,
#         #         padding=p,
#         #     ),
#         #     nn.Sigmoid(),
#         #     # Output: N x channels_img x 512 x 512
#         # )

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding, oDims):
#         return nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#             ),
#             # self.activation(),
#             nn.LeakyReLU(0.2),
#             # nn.InstanceNorm2d(out_channels, affine=True),
#             nn.LayerNorm([out_channels, oDims[0], oDims[1]], elementwise_affine=True),
#             InitNorm(),
#             # nn.BatchNorm2d(out_channels, affine=False),
#             # ContainNorm(),
#             # nn.Dropout(0.1),
#         )

#     def forward(self, z):
#         #############################
#             # SAVE
#         #############################
#         # n = z.shape[0]
#         # init_channels = 4
#         # init = z[:, 0:init_channels*16**2].reshape(n, init_channels, 16, 16)  # set initial frame to first chunk of seed

#         # z = z[:, init_channels*16**2:]  # delete init section from seed
#         # z = z.view(n, -1, self.k, self.k)  # reshape remaining seed into kernels
#         # kern1 = z[:, 0:0+16]
#         # kern2 = z[:, 16:16+16]
#         # kern3 = z[:, 32:32+16]
#         # # kern4 = z[:, 24:]
#         # # kern5 = z[:, 34:]
#         #############################

#         n = z.shape[0]
#         init_channels = 1
#         # init = torch.normal(
#         #     torch.zeros([n, init_channels, 16, 16]),
#         #     torch.ones([n, init_channels, 16, 16]),
#         # ).to(z.device)
#         init = torch.ones([n, init_channels, 2, 2]).to(z.device)

#         z = z.view(n, -1, self.k, self.k)  # reshape seed into kernels
#         # kernA = z[:, 0:0+16]
#         # kernB = z[:, 16:16+16]
#         # kernC = z[:, 32:32+16]
#         # kern0 = z[:, 48:48+16]
#         # kern1 = z[:, 64:64+16]
#         # kern2 = z[:, 80:80+16]
#         # kern3 = z[:, 96:]

#         z_kernels = []
#         for i in range(14):
#             z_kernels.append(z[:, i * self.num_zk: (i+1) * self.num_zk])

#         # inject kernels at beginning of each resolution layer
#         x = self.injection_blockA(init, z_kernels[0])
#         x = self.injection_blockA2(x, z_kernels[1])
#         x = self.lA(x)

#         x = self.injection_blockB(x, z_kernels[2])
#         x = self.injection_blockB2(x, z_kernels[3])
#         x = self.lB(x)

#         x = self.injection_blockC(x, z_kernels[4])
#         x = self.injection_blockC2(x, z_kernels[5])
#         x = self.lC(x)

#         x = self.injection_block0(x, z_kernels[6])
#         x = self.injection_block02(x, z_kernels[7])
#         x = self.l0(x)

#         x = self.injection_block1(x, z_kernels[8])
#         x = self.injection_block12(x, z_kernels[9])
#         x = self.l1(x)

#         x = self.injection_block2(x, z_kernels[10])
#         x = self.injection_block22(x, z_kernels[11])
#         x = self.l2(x)

#         x = self.injection_block3(x, z_kernels[12])
#         x = self.injection_block32(x, z_kernels[13])
#         x = self.l3(x)

#         # x = self.injection_block4(x, kern4)
#         # x = self.l4(x)

#         # x = self.injection_block5(x, kern5)
#         # x = self.l5(x)
#         return x


class Decoder(nn.Module):
    def __init__(self, z_dim, channels_img, features_gen=16, activation=nn.ReLU):
        super(Decoder, self).__init__()
        self.activation = activation
        k = 7
        s = 1
        p = 3
        num_zk = 1
        self.num_zk = num_zk
        zk = 8
        self.zk = zk
        self.k = k
        self.s = s
        self.p = p
        self.lA = nn.Sequential(self._block(num_zk, features_gen, k, s, p, (8, 8)))
        self.lB = nn.Sequential(self._block(num_zk, features_gen, k, s, p, (8, 8)))
        self.lC = nn.Sequential(
            self._block(num_zk, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
        )
        self.lD = nn.Sequential(
            self._block(num_zk, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
        )
        self.lE = nn.Sequential(
            self._block(num_zk, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
            self._block(features_gen, features_gen, k, s, p, (16, 16)),
            nn.Upsample((32, 32), mode="bilinear"),  # 16x16 => 32x32
        )
        self.lF = nn.Sequential(
            self._block(num_zk, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
            self._block(features_gen, features_gen, k, s, p, (16, 16)),
            nn.Upsample((32, 32), mode="bilinear"),  # 16x16 => 32x32
        )
        self.lG = nn.Sequential(
            self._block(num_zk, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
            self._block(features_gen, features_gen, k, s, p, (16, 16)),
            nn.Upsample((32, 32), mode="bilinear"),  # 16x16 => 32x32
            self._block(features_gen, features_gen, k, s, p, (32, 32)),
            nn.Upsample((64, 64), mode="bilinear"),  # 32x32 => 64x64
        )
        self.lH = nn.Sequential(
            self._block(num_zk, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
            self._block(features_gen, features_gen, k, s, p, (16, 16)),
            nn.Upsample((32, 32), mode="bilinear"),  # 16x16 => 32x32
            self._block(features_gen, features_gen, k, s, p, (32, 32)),
            nn.Upsample((64, 64), mode="bilinear"),  # 32x32 => 64x64
        )
        # self.lI = nn.Sequential(
        #     self._block(num_zk, features_gen, k, s, p, (8, 8)),
        #     nn.Upsample((16, 16), mode="bilinear"), # 8x8 => 16x16
        #     self._block(features_gen, features_gen, k, s, p, (16, 16)),
        #     nn.Upsample((32, 32), mode="bilinear"), # 16x16 => 32x32
        #     self._block(features_gen, features_gen, k, s, p, (32, 32)),
        #     nn.Upsample((64, 64), mode="bilinear"), # 32x32 => 64x64
        #     self._block(features_gen, features_gen, k, s, p, (64, 64)),
        #     nn.Upsample((128, 128), mode="bilinear"), # 64x64 => 128x128
        # )
        self.l0 = nn.Sequential(
            self._block(features_gen, features_gen, k, s, p, (8, 8))
        )
        self.l1 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (8, 8)),
            nn.Upsample((16, 16), mode="bilinear"),  # 8x8 => 16x16
        )
        self.l2 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (16, 16))
        )
        self.l3 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (16, 16)),
            nn.Upsample((32, 32), mode="bilinear"),  # 16x16 => 32x32
        )
        self.l4 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (32, 32))
        )
        self.l5 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (32, 32)),
            nn.Upsample((64, 64), mode="bilinear"),  # 32x32 => 64x64
        )
        self.l6 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (64, 64))
        )
        self.l7 = nn.Sequential(
            self._block(features_gen * 2, features_gen, k, s, p, (64, 64)),
            nn.Upsample((128, 128), mode="bilinear"),  # 64x64 => 128x128
        )
        # self.l8 = nn.Sequential(
        #     self._block(features_gen*2, features_gen, k, s, p, (128, 128)),
        #     nn.Upsample((256, 256), mode="bilinear"), # 128x128 => 256x256
        # )
        self.finish = nn.Sequential(
            # self._block(features_gen, features_gen, k, s, p, (256, 256)),
            nn.Conv2d(features_gen, channels_img, kernel_size=k, stride=s, padding=p),
            InitNorm(0, 0.25),
            nn.Sigmoid(),
            ## Output: N x channels_img x 128 x 128
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, oDims):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            # self.activation(),
            nn.LeakyReLU(0.2),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LayerNorm([out_channels, oDims[0], oDims[1]], elementwise_affine=True),
            # InitNorm(constrain_mean = False),
            # nn.BatchNorm2d(out_channels, affine=False),
            # ContainNorm(),
            # nn.Dropout(0.1),
        )

    def forward(self, z):
        n = z.shape[0]
        z = z.view(n, -1, self.zk, self.zk)  # reshape seed into kernels
        z_kernels = []
        for i in range(8):
            z_kernels.append(z[:, i * self.num_zk : (i + 1) * self.num_zk])
        # inject kernels at beginning of each resolution layer
        x = self.lA(z_kernels[0])
        x = self.l0(x)
        x = torch.cat((x, self.lB(z_kernels[1])), 1)
        x = self.l1(x)
        x = torch.cat((x, self.lC(z_kernels[2])), 1)
        x = self.l2(x)
        x = torch.cat((x, self.lD(z_kernels[3])), 1)
        x = self.l3(x)
        x = torch.cat((x, self.lE(z_kernels[4])), 1)
        x = self.l4(x)
        x = torch.cat((x, self.lF(z_kernels[5])), 1)
        x = self.l5(x)
        x = torch.cat((x, self.lG(z_kernels[6])), 1)
        x = self.l6(x)
        x = torch.cat((x, self.lH(z_kernels[7])), 1)
        x = self.l7(x)
        # x = torch.cat((x, self.lI(z_kernels[8])), 1)
        # x = self.l8(x)
        x = self.finish(x)
        return x


def linsig(x):
    # sigmoid that is extremely linear on the interval between -interval and interval.
    interval = 4
    c = (
        torch.Tensor([4.32790682747]).to(x.device) * interval
    )  # important constant for high linearity.
    return c * torch.sigmoid(x / interval) - c / 2


class DAE(nn.Module):
    def __init__(
        self,
        z_dim=100,
        channels_img=1,
        features_enc=16,
        features_dec=16,
        activation=nn.ReLU,
    ):
        super(DAE, self).__init__()

        self.z_dim = z_dim
        self.activation = activation

        k = 7
        s = 1
        p = 3

        self.k = k
        self.s = s
        self.p = p

        num_zk = 1
        self.num_zk = num_zk
        zk = 8
        self.zk = zk

        # self.zk0 = nn.Sequential(
        #     self._block(channels_img,  features_enc, k, s, p, (256, 256)),
        #     Interpolate((128,128), mode="bilinear"),
        #     self._block(features_enc,  features_enc, k, s, p, (128, 128)),
        #     Interpolate((64,64), mode="bilinear"),
        #     self._block(features_enc, features_enc, k, s, p, (64, 64)),
        #     Interpolate((32,32), mode="bilinear"),
        #     self._block(features_enc, features_enc, k, s, p, (32, 32)),
        #     Interpolate((16,16), mode="bilinear"),
        #     self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        # )

        # self.c1 = nn.Sequential(
        #     # Interpolate((256,256), mode="bilinear"),
        #     self._block(channels_img, features_enc, k, s, p, (256, 256)),
        #     # self._block(features_enc, features_enc, k, s, p, (256, 256)),
        # )

        self.zk1 = nn.Sequential(
            # self._block(features_enc,  features_enc, k, s, p, (256, 256)),
            # Interpolate((128,128), mode="bilinear"),
            self._block(channels_img, features_enc, k, s, p, (128, 128)),
            Interpolate((64, 64), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (64, 64)),
            Interpolate((32, 32), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (32, 32)),
            Interpolate((16, 16), mode="bilinear"),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c2 = nn.Sequential(
            # Interpolate((128,128), mode="bilinear"),
            self._block(channels_img, features_enc, k, s, p, (128, 128)),
            # self._block(features_enc, features_enc, k, s, p, (128, 128)),
        )
        self.zk2 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (128, 128)),
            Interpolate((64, 64), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (64, 64)),
            Interpolate((32, 32), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (32, 32)),
            Interpolate((16, 16), mode="bilinear"),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c3 = nn.Sequential(
            Interpolate((64, 64), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (64, 64)),
            # self._block(features_enc, features_enc, k, s, p, (64, 64)),
        )
        self.zk3 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (64, 64)),
            Interpolate((32, 32), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (32, 32)),
            Interpolate((16, 16), mode="bilinear"),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c4 = nn.Sequential(
            Interpolate((32, 32), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (32, 32)),
            # self._block(features_enc, features_enc, k, s, p, (32, 32)),
        )
        self.zk4 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (32, 32)),
            Interpolate((16, 16), mode="bilinear"),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c5 = nn.Sequential(
            Interpolate((16, 16), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (16, 16)),
            # self._block(features_enc, features_enc, k, s, p, (16, 16)),
        )
        self.zk5 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (16, 16)),
            # self._block(features_enc, features_enc, k, s, p, (16, 16)),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c6 = nn.Sequential(
            Interpolate((8, 8), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (8, 8)),
            # self._block(features_enc, features_enc, k, s, p, (8, 8)),
        )
        self.zk6 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (8, 8)),
            # self._block(features_enc, features_enc, k, s, p, (8, 8)),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c7 = nn.Sequential(
            Interpolate((4, 4), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (4, 4)),
            # self._block(features_enc, features_enc, k, s, p, (4, 4)),
        )
        self.zk7 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (4, 4)),
            # self._block(features_enc, features_enc, k, s, p, (4, 4)),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        self.c8 = nn.Sequential(
            Interpolate((2, 2), mode="bilinear"),
            self._block(features_enc, features_enc, k, s, p, (2, 2)),
            # self._block(features_enc, features_enc, k, s, p, (2, 2)),
        )
        self.zk8 = nn.Sequential(
            self._block(features_enc, features_enc, k, s, p, (2, 2)),
            # self._block(features_enc, features_enc, k, s, p, (2, 2)),
            self._z_block(features_enc, num_zk, k, s, p, (zk, zk)),
        )

        # self.Z_Normalizer = nn.LayerNorm([z_dim], elementwise_affine=False)
        self.Z_Containment = ContainNorm()

        self.decoder = Decoder(z_dim, channels_img, features_dec)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, oDims):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(0.2),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LayerNorm([out_channels, oDims[0], oDims[1]], elementwise_affine=True),
            # InitNorm(constrain_mean = False),
            # nn.BatchNorm2d(out_channels, affine=False),
            # ContainNorm(),
            # nn.Dropout(0.1),
        )

    def _z_block(self, in_channels, out_channels, kernel_size, stride, padding, oDims):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            Interpolate(oDims, mode="bilinear"),
            nn.LayerNorm([out_channels, *oDims], elementwise_affine=False),
            # nn.InstanceNorm2d(out_channels, affine=False),
            InitNorm(constrain_mean=False),
            # ContainNorm(),
            # nn.BatchNorm2d(out_channels, affine=False),
        )

    def encode(self, x, include_uncut=False):
        # x = self.c0(x)
        # z0 = self.zk0(x).flatten(1)

        # x = self.c1(x)
        z1 = self.zk1(x).flatten(1)

        x = self.c2(x)
        z2 = self.zk2(x).flatten(1)

        x = self.c3(x)
        z3 = self.zk3(x).flatten(1)

        x = self.c4(x)
        z4 = self.zk4(x).flatten(1)

        x = self.c5(x)
        z5 = self.zk5(x).flatten(1)

        x = self.c6(x)
        z6 = self.zk6(x).flatten(1)

        x = self.c7(x)
        z7 = self.zk7(x).flatten(1)

        x = self.c8(x)
        z8 = self.zk8(x).flatten(1)

        uncut_z = torch.cat((z8, z7, z6, z5, z4, z3, z2, z1), 1)

        # uncut_z = self.Z_Containment(uncut_z)

        if include_uncut:
            # return linsig(uncut_z), uncut_z
            return uncut_z, uncut_z
        else:
            # return linsig(uncut_z)
            return uncut_z

    def forward(self, x, include_uncut=False):
        if include_uncut:
            z, uncut_z = self.encode(x, include_uncut=True)
            return self.decoder(z), z, uncut_z
        else:
            z = self.encode(x)
            return self.decoder(z), z


class CodeCritic(nn.Module):
    def __init__(self, z_dim):

        super(CodeCritic, self).__init__()

        self.disc = nn.Sequential(
            self._block(z_dim, 2 * z_dim),
            self._block(2 * z_dim, 2 * z_dim),
            self._block(2 * z_dim, 2 * z_dim),
            self._block(2 * z_dim, 2 * z_dim),
            self._block(2 * z_dim, 2 * z_dim),
            self._block(2 * z_dim, 2 * z_dim),
            nn.Linear(2 * z_dim, 1),
        )

    def _block(self, iDim, oDim):
        return nn.Sequential(
            nn.Linear(iDim, oDim),
            # nn.InstanceNorm1d(oDim, affine=True),
            # nn.LayerNorm([oDim], elementwise_affine=True),
            # nn.BatchNorm1d(oDim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Critic(nn.Module):
    def __init__(self, channels_img, features_d):

        super(Critic, self).__init__()

        k = 7
        s = 1
        p = 3

        self.disc = nn.Sequential(
            # input: N x channels_img x 512 x 512
            # self._block(channels_img, features_d, k, s, p, 512, 512),
            # self._block(features_d, features_d, k, s, p, 512, 512),
            # self._block(features_d, features_d, k, s, p, 512, 512),
            # Interpolate((256,256), mode="bilinear"),
            # self._block(features_d, features_d, k, s, p, 256, 256),
            # # self._block(features_d, features_d, k, s, p, 256, 256),
            # self._block(features_d, features_d, k, s, p, 256, 256),
            # Interpolate((128,128), mode="bicubic"),
            self._block(channels_img, features_d, k, s, p, 128, 128),
            # self._block(features_d, features_d, k, s, p, 128, 128),
            self._block(features_d, features_d, k, s, p, 128, 128),
            Interpolate((64, 64), mode="bicubic"),
            self._block(features_d, features_d, k, s, p, 64, 64),
            # self._block(features_d, features_d, k, s, p, 64, 64),
            self._block(features_d, features_d, k, s, p, 64, 64),
            Interpolate((32, 32), mode="bicubic"),
            self._block(features_d, features_d, k, s, p, 32, 32),
            # self._block(features_d, features_d, k, s, p, 32, 32),
            self._block(features_d, features_d, k, s, p, 32, 32),
            Interpolate((16, 16), mode="bicubic"),
            self._block(features_d, features_d, k, s, p, 16, 16),
            # self._block(features_d, features_d, k, s, p, 16, 16),
            self._block(features_d, features_d, k, s, p, 16, 16),
            nn.Conv2d(features_d, 1, 16, 1, 0, bias=True),
            # nn.Flatten(1),
            # nn.LazyLinear(1),
        )

    def _block(
        self, in_channels, out_channels, kernel_size, stride, padding, oDim1, oDim2
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=True
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            # nn.LayerNorm([out_channels, oDim1, oDim2], elementwise_affine=True),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


def initialize_normal(model):
    """weird flex but okay"""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif isinstance(
            m,
            (
                nn.BatchNorm2d,
                nn.BatchNorm1d,
                nn.LayerNorm,
                nn.InstanceNorm2d,
                nn.InstanceNorm1d,
            ),
        ):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def initialize_kaiming_he(model, a=0, nonlinearity: str = "leaky_relu", mode="fan_in"):
    """Kaiming He initialization. See: https://arxiv.org/pdf/1502.01852.pdf"""
    for m in model.modules():
        ### Kaiming He: ###
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            nn.init.constant_(m.bias, 0)
        elif isinstance(
            m,
            (
                nn.BatchNorm2d,
                nn.BatchNorm1d,
                nn.LayerNorm,
                nn.InstanceNorm2d,
                nn.InstanceNorm1d,
            ),
        ):
            try:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            except Exception:
                print("no weights or no biases")
                pass


def initialize_kaiming_he_big(
    model, c=10, a=0, nonlinearity: str = "leaky_relu", mode="fan_in"
):
    """Kaiming He initialization. See: https://arxiv.org/pdf/1502.01852.pdf"""
    for m in model.modules():
        ### Kaiming He: ###
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            m.weight.data *= c
            nn.init.constant_(m.bias, 0)
        elif isinstance(
            m,
            (
                nn.BatchNorm2d,
                nn.BatchNorm1d,
                nn.LayerNorm,
                nn.InstanceNorm2d,
                nn.InstanceNorm1d,
            ),
        ):
            try:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            except Exception:
                print("no weights or no biases")
                pass


def initialize_kaiming_zero(
    model, a=0, nonlinearity: str = "leaky_relu", mode="fan_in"
):
    """Kaiming He initialization. See: https://arxiv.org/pdf/1502.01852.pdf"""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            m.weight.data /= 4
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def initialize_clear(model):
    for m in model.modules():
        print("\nm:", m)
        try:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            print(f"fan_in: {fan_in}, fan_out: {fan_out}")
            maxnodes = 1568
        except:
            continue
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.constant_(m.weight, -1 / fan_in)
            m.weight.data.flatten()[
                torch.multinomial(
                    torch.ones_like(m.weight.flatten()), m.weight.numel() // 2
                )
            ] = (1 / fan_in)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


import scipy
from pprint import pprint


def bce_loss(y_out, y):
    return nn.functional.binary_cross_entropy(y_out, y, reduction="mean")


def gabian_distance_scipy(z):
    """finds the Gabian Distance between a 1d vector and a standardized gaussian distribution
    (mean=0 and sigma=1)"""

    norm = scipy.stats.norm()
    repZ = scipy.stats.gaussian_kde(
        z.flatten(0).detach().to("cpu").numpy(),
        bw_method=scipy.stats.gaussian_kde.silvermans_factor,
    )

    overlap, err = scipy.integrate.quad(lambda x: min(norm.pdf(x), repZ(x)), -5, 5)
    return (1 - overlap).pow(2)


def depthwise_skewness_fn(x):
    std, mean = torch.std_mean(x, 0)
    n = torch.Tensor([x.shape[0]]).to(x.device)
    eps = 5e-3  # for stability

    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    skewness = sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(0).T).T.pow(3), 0) / n)
        / std.pow(3).clamp(min=eps)
    )
    return skewness


def depthwise_kurtosis_fn(x):
    std, mean = torch.std_mean(x, 0)
    n = torch.Tensor([x.shape[0]]).to(x.device)
    eps = 5e-3  # for stability

    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    kurtosis = sample_bias_adjustment * (
        (n + 1)
        * (
            (torch.sum((z.T - mean.unsqueeze(0).T).T.pow(4), 0) / n)
            / std.pow(4).clamp(min=eps)
        )
        - 3 * (n - 1)
    )
    return kurtosis


def depthwise_bimodality_coefficient(x):
    """
    The logic behind this coefficient is that a bimodal (or multimodal) distribution with light 
    tails will have very low kurtosis, an asymmetric character, or both – all of which increase this
    coefficient.
    The smaller this value is the more likely the data are to follow a unimodal distribution.
    As a rule: if return value ≤ 0.555 (BC value for uniform distribution), the data are considered
    to follow a unimodal distribution. Otherwise, they follow a bimodal or multimodal distribution.
    """
    std, mean = torch.std_mean(x, 0)
    n = torch.Tensor([x.shape[0]]).to(x.device)
    eps = 5e-3  # for stability

    # calculate skewness:
    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    skew = sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(0).T).T.pow(3), 0) / n)
        / std.pow(3).clamp(min=eps)
    )

    # calculate kurtosis:
    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    kurt = sample_bias_adjustment * (
        (n + 1)
        * (
            (torch.sum((z.T - mean.unsqueeze(0).T).T.pow(4), 0) / n)
            / std.pow(4).clamp(min=eps)
        )
        - 3 * (n - 1)
    )

    # calculate bimodality coefficient:
    BC = (skew.pow(2) + 1) / (kurt + 3 * ((n - 2).pow(2) / ((n - 2) * (n - 3))))

    return BC


def modularity_loss(x):
    bcd = torch.mean(depthwise_bimodality_coefficient(x))
    return F.binary_cross_entropy(bcd, torch.zeros_like(bcd).to(x.device))


def node_stat_loss(z, target_mean=0, target_std=1):
    n = z.shape[0]
    assert n > 1

    σ, μ = torch.std_mean(z, 0)
    # print([f"node {idx} μ={μ[idx].item()}, σ={σ[idx].item()}" for idx in range(0, μ.shape[0], 256)])
    μ = μ - target_mean
    p_var = (σ / target_std) ** 2
    KLD = -0.5 * torch.mean(1 + (p_var.pow(2)).log() - μ.pow(2) - p_var.pow(2))

    eps = 5e-3  # for stability

    # compute average skewness
    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    # using custom skew estimation because we do not want negative and positive skews to cancel.
    abs_skew = sample_bias_adjustment * torch.mean(
        (torch.sum((z.T - μ.unsqueeze(0).T).T.abs().pow(3), 0) / n)
        / σ.pow(3).clamp(min=eps)
    )

    # compute average kurtosis
    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    # using custom kurtosis estimation to aviod recomputation of moments.
    kurtosis = sample_bias_adjustment * torch.mean(
        (n + 1)
        * (
            (torch.sum((z.T - μ.unsqueeze(0).T).T.pow(4), 0) / n)
            / σ.pow(4).clamp(min=eps)
        )
        - 3 * (n - 1)
    )

    return KLD + abs_skew + kurtosis


def image_gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, C, H, W = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, W = real.shape
    alpha = torch.rand((batch_size, 1)).repeat(1, W).to(device)
    interpolated_codes = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_codes)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_codes,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model=torchvision.models.efficientnet_b0, resize=True, depth=3):
        super(PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(model(pretrained=True).features[:4].eval())
        if depth >= 2:
            blocks.append(model(pretrained=True).features[4:9].eval())
        if depth >= 3:
            blocks.append(model(pretrained=True).features[9:16].eval())
        if depth >= 4:
            blocks.append(model(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        batch_size = in_.shape[0]
        if in_.shape[1] != 3:
            in_ = in_.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        in_ = (in_ - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            in_ = self.transform(
                in_, mode="bicubic", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bicubic", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = in_
        y = target
        losses = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            if i in feature_layers:
                losses.append(torch.nn.functional.mse_loss(x, y, reduction="mean"))
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                losses.append(
                    torch.nn.functional.mse_loss(gram_x, gram_y, reduction="mean")
                )
        return losses


# Weighted Delta Accumulator
class Loss_Multiplex(object):
    def __init__(
        self,
        goal_functions: list,
        scale_factors,
        num_losses: None,
        log_length: int = 200,
        base_window: int = 15,
        window_step: int = 5,
        loss_weight_pursuit_rate: float = 1 / 10,
        auto_pursuit: bool = True,
        pool_size: int = 1,
    ):
        self.step = 1
        self.pool_size = pool_size
        self.auto_pursuit = auto_pursuit
        assert type(scale_factors) in (list, tuple) or scale_factors == "auto"
        if num_losses is None:
            self.num_losses = len(goal_functions)
        else:
            self.num_losses = num_losses
        if scale_factors != "auto":
            try:
                assert len(scale_factors) == num_losses
            except AssertionError:
                raise AssertionError(
                    "Number of scale factors does not match number of losses."
                )
        self.scale_factors = scale_factors
        self.goal_functions = goal_functions
        self.num_losses = num_losses  # not necessarily the same as len(goal_functions)
        # ^ (some goal functions return more than one loss)
        self.weights = [1 / num_losses for _ in range(num_losses)]
        self.loss_weight_pursuit_rate = loss_weight_pursuit_rate

        self.log_length = log_length
        self.base_window = base_window
        self.window_step = window_step

        if self.pool_size > 1:
            self.loss_histories = [[0] for _ in range(num_losses)]
        else:
            self.loss_histories = [[] for _ in range(num_losses)]

        self.prev_multiloss = 0

    def calc_weight(self, x, total):
        try:
            y = x / total
        except ZeroDivisionError:
            y = 1 / self.num_losses
        return y

    def tupify(self, seq):
        """wrap non-iterable items of sequence in their own tuple."""
        tupseq = []
        for item in seq:
            if isinstance(item, (list, tuple)):
                tupseq.append(item)
            else:
                tupseq.append((item,))
        return tupseq

    # def loss(self, y_out, y, μ, logvar, criticism_recon, criticism_gen, code_criticism):
    def loss(self, *data, step_weights: bool = False):
        args = self.tupify(
            data
        )  # make sure each datum is iterable for proper unpacking
        losses = [x(*y) for x, y in zip(self.goal_functions, args)]
        losses = recursive_unpack(
            losses
        )  # some goal functions return tuples of more than one loss
        if self.step == 1 and self.scale_factors == "auto":
            self.scale_factors = [1 / x.item() for x in losses]
            print("\nself.scaled_factors:", self.scale_factors)
        scaled_losses = [
            x * y for x, y in zip(losses, self.scale_factors)
        ]  # some goal functions return tuples of more than one loss

        for idx, l in enumerate(losses):
            if self.step % self.pool_size == 0:
                self.loss_histories[idx].append(l.item() / self.pool_size)
            else:
                self.loss_histories[idx][-1] += l.item() / self.pool_size

        if len(self.loss_histories[0]) > self.log_length:
            for hist in self.loss_histories:
                hist.pop(0)

        try:
            prev_losses = [hist[-2] for hist in self.loss_histories]
        except IndexError:
            prev_losses = [0 for _ in range(self.num_losses)]

        deltas = [y - x for x, y in zip(prev_losses, losses)]
        scaled_deltas = [x * y for x, y in zip(deltas, self.scale_factors)]
        weighted_deltas = [x * y for x, y in zip(scaled_deltas, self.weights)]

        multiplex_loss = self.prev_multiloss + sum(weighted_deltas)
        self.prev_multiloss = multiplex_loss.detach()

        prev_weights = self.weights
        if (self.auto_pursuit and self.step % self.pool_size == 0) or step_weights:
            self.weight_step()

        self.step += 1

        return (
            multiplex_loss,
            [x.item() for x in losses],
            [x.item() for x in scaled_losses],
            [(x * y).item() for x, y in zip(scaled_losses, prev_weights)],
            prev_weights,
        )

    def weight_step(self):
        # calculating weights for next step; (the elegant and correct (maybe?) way)
        if len(self.loss_histories[0]) < self.base_window:
            return
        scaled_loss_histories = []
        for history, scale in zip(self.loss_histories, self.scale_factors):
            scaled_loss_histories.append([x * scale for x in history])
        prev_weights = self.weights
        if len(self.loss_histories[0]) >= self.base_window:
            slews = [0 for _ in self.loss_histories]
            for i, hist in enumerate(scaled_loss_histories):
                # find window over which to estimate slew rate:
                # (watches for time proportional to noise in signal)
                window = self.base_window
                while window <= len(hist):
                    line = linregress(range(window), hist[-window:])
                    if line.stderr < 5e-3:
                        break
                    else:
                        window += self.window_step
                # print(i, "window:", window)
                slews[i] = abs(line.slope)
            sum_slews = sum(slews)

            eph_weights = [
                self.calc_weight(x, sum_slews) for x in slews
            ]  # IMPORTANT: this must be done only after calculation of multiplex_loss.
            diffs = [x - y for x, y in zip(eph_weights, self.weights)]
            epsilon = 1e-5
            self.weights = [
                x + max(self.loss_weight_pursuit_rate * dif, min(dif, epsilon))
                for x, dif in zip(self.weights, diffs)
            ]


# Weighted Rect-Delta Accumulator
# class Loss_Multiplex(object):
#     def __init__(
#             self,
#             goal_functions: list,
#             scale_factors: list,
#             num_losses: None,
#             log_length: int=200,
#             base_window: int=15,
#             window_step: int=5,
#             loss_weight_pursuit_rate: float=1/10,
#             auto_pursuit: bool=True,
#             pool_size: int=1,
#         ):
#         self.step = 1
#         self.pool_size = pool_size
#         self.auto_pursuit = auto_pursuit
#         if num_losses is None:
#             self.num_losses = len(goal_functions)
#         else:
#             self.num_losses = num_losses
#         try:
#             assert len(scale_factors) == num_losses
#         except AssertionError:
#             raise AssertionError("Number of scale factors does not match number of losses.")
#         self.scale_factors = scale_factors
#         self.goal_functions = goal_functions
#         self.num_losses = num_losses # not necessarily the same as len(goal_functions)
#         # ^ (some goal functions return more than one loss)
#         self.weights = [1/num_losses for _ in range(num_losses)]
#         self.loss_weight_pursuit_rate = loss_weight_pursuit_rate

#         self.log_length = log_length
#         self.base_window = base_window
#         self.window_step = window_step

#         if self.pool_size > 1:
#             self.loss_histories = [[0,] for _ in range(num_losses)]
#         else:
#             self.loss_histories = [[] for _ in range(num_losses)]

#         self.prev_multiloss = 0

#     def calc_weight(self, x, total):
#         try:
#             y = x/total
#         except ZeroDivisionError:
#             y = 1/self.num_losses
#         return y

#     def tupify(self, seq):
#         '''wrap non-iterable items of sequence in their own tuple.'''
#         tupseq = []
#         for item in seq:
#             if isinstance(item, (list, tuple)):
#                 tupseq.append(item)
#             else:
#                 tupseq.append((item,))
#         return tupseq

#     # def loss(self, y_out, y, μ, logvar, criticism_recon, criticism_gen, code_criticism):
#     def loss(self, *data, step_weights: bool=False):
#         args = self.tupify(data)  # make sure each datum is iterable for proper unpacking
#         losses = [x(*y) for x, y in zip(self.goal_functions, args)]
#         losses = recursive_unpack(losses)  # some goal functions return tuples of more than one loss
#         scaled_losses = [x * y for x, y in zip(losses, self.scale_factors)]  # some goal functions return tuples of more than one loss

#         for idx, l in enumerate(losses):
#             if self.step % self.pool_size == 0:
#                 self.loss_histories[idx].append(l.item() / self.pool_size)
#             else:
#                 self.loss_histories[idx][-1] += l.item() / self.pool_size

#         if len(self.loss_histories[0]) > self.log_length:
#             for hist in self.loss_histories:
#                 hist.pop(0)

#         try:
#             prev_losses = [hist[-2] for hist in self.loss_histories]
#         except IndexError:
#             prev_losses = [0 for _ in range(self.num_losses)]

#         deltas = [y - x for x, y in zip(prev_losses, losses)]
#         scaled_deltas = [x * y for x, y in zip(deltas, self.scale_factors)]
#         weighted_deltas = [x * y for x, y in zip(scaled_deltas, self.weights)]

#         multiplex_loss = self.prev_multiloss + sum(weighted_deltas)
#         self.prev_multiloss = multiplex_loss.detach()

#         prev_weights = self.weights
#         if (self.auto_pursuit and self.step % self.pool_size == 0) or step_weights:
#             self.weight_step()

#         self.step += 1

#         return multiplex_loss, [x.item() for x in losses], [x.item() for x in scaled_losses], [(x * y).item() for x, y in zip(scaled_losses, prev_weights)], prev_weights

#     def weight_step(self):
#         # calculating weights for next step; (the elegant and correct (maybe?) way)
#         if len(self.loss_histories[0]) < self.base_window:
#             return
#         scaled_loss_histories = []
#         for history, scale in zip(self.loss_histories, self.scale_factors):
#             scaled_loss_histories.append([x * scale for x in history])
#         prev_weights = self.weights
#         if len(self.loss_histories[0]) >= self.base_window:
#             slopes = [0 for _ in self.loss_histories]
#             for i, hist in enumerate(scaled_loss_histories):
#                 # find window over which to estimate slopes:
#                 # (watches for time proportional to noise in signal)
#                 window = self.base_window
#                 while window <= len(hist):
#                     line = linregress(range(window), hist[-window:])
#                     if line.stderr < 5e-3:
#                         break
#                     else:
#                         window += self.window_step
#                 # print(i, "window:", window)
#                 slopes[i] = line.slope
#             max_slope = max(slopes)
#             slopes = [-(x - max_slope) for x in slopes]
#             sum_slopes = sum(slopes)

#             eph_weights = [self.calc_weight(x, sum_slopes) for x in slopes]  # IMPORTANT: this must be done only after calculation of multiplex_loss.
#             diffs = [x - y for x, y in zip(eph_weights, self.weights)]
#             epsilon = 1e-5
#             self.weights = [x + max(self.loss_weight_pursuit_rate*dif, min(dif, epsilon)) for x, dif in zip(self.weights, diffs)]

from torch.distributions import MultivariateNormal, Normal
from torch.distributions.cauchy import Cauchy as Cauchy


def gabian_pdf(data, x_tics=None, bandwidth_scale=1, depth_limit=True):
    """estimates the Gabian Probability Density Function of a batch of 1d vectors"""
    chunksize = 64

    def kde_prob(
        data,
        x,
        kernel=Normal(loc=0, scale=1),
        bandwidth_scale=1,
        chunksize=chunksize,
        depth_limit=True,
    ):
        """This function is memory intensive"""
        if depth_limit:
            z = data[:chunksize].flatten(0)
        else:
            z = data.flatten(0)
        silvermans_factor = ((4 * torch.std(z).pow(5)) / (3 * z.numel())).pow(1 / 5)
        # silvermans_factor = (z.numel() * 3/4)**(-1/5)

        bw = silvermans_factor * bandwidth_scale
        try:
            a = (z.unsqueeze(1) - x) / bw
            a = kernel.log_prob(a)
            a = torch.exp(a)
            a = bw ** (-1) * a
            a = a.sum(dim=0)
            prob = a / z.numel()
        except Exception as e:
            print(x)
            raise e
        return prob

    if x_tics is None:
        steps = 256  # steps for estimation
        range = 9
        if depth_limit:
            a = max(
                torch.min(data[:chunksize]).item(), -range / 2
            )  # lower integration bound
            b = min(
                torch.max(data[:chunksize]).item(), range / 2
            )  # upper integration bound
        else:
            a = max(torch.min(data).item(), -range / 2)  # lower integration bound
            b = min(torch.max(data).item(), range / 2)  # upper integration bound
        x_tics = torch.Tensor(np.linspace(a, b, steps)).to(data.device)
    else:
        assert isinstance(x_tics, torch.Tensor)
        x_tics = x_tics.to(data.device)
    x_tics.requires_grad = True
    kde_y_tics = kde_prob(
        data, x_tics, bandwidth_scale=bandwidth_scale, depth_limit=depth_limit
    )
    return kde_y_tics


def gabian_distance(
    data, x_tics=None, target_mean=0, target_std=1, bandwidth_scale=1, depth_limit=True
):
    """finds the Gabian Distance between a batch of 1d vectors and a batch of standard gaussian
    distributions (mean=0 and sigma=1)"""
    chunksize = 64

    def kde_prob(
        data,
        x,
        kernel=Normal(loc=0, scale=1),
        bandwidth_scale=1,
        chunksize=chunksize,
        depth_limit=True,
    ):
        """This function is memory intensive"""
        # # in this case chunk_size determines number of batches to take at once.
        # total_elements = data.numel()
        # weighted_avg_prob = torch.zeros_like(x)
        # for chunk in data.split(chunksize):
        #     z = chunk.flatten(0)
        #     silvermans_factor = ((4*torch.std(z).pow(5))/(3*z.numel())).pow(1/5)
        #     bw = silvermans_factor * bandwidth_scale
        #     try:
        #         a = (z.unsqueeze(1) - x) / bw
        #         a = kernel.log_prob(a)
        #         a = torch.exp(a)
        #         a = bw**(-1) * a
        #         a = a.sum(dim=0)
        #         prob = a / z.numel()
        #         weighted_avg_prob += prob * (z.numel() / total_elements)
        #     except Exception as e:
        #         print(x)
        #         raise e
        # return weighted_avg_prob
        # # in this case chunk_size determines number of batches to process.
        if depth_limit:
            z = data[:chunksize].flatten(0)
        else:
            z = data.flatten(0)
        silvermans_factor = ((4 * torch.std(z).pow(5)) / (3 * z.numel())).pow(1 / 5)
        bw = silvermans_factor * bandwidth_scale
        try:
            a = (z.unsqueeze(1) - x) / bw
            a = kernel.log_prob(a)
            a = torch.exp(a)
            a = bw ** (-1) * a
            a = a.sum(dim=0)
            prob = a / z.numel()
        except Exception as e:
            print(x)
            raise e
        return prob

    if x_tics is None:
        steps = 256  # steps for estimation
        range = 9
        if depth_limit:
            a = max(
                torch.min(data[:chunksize]).item(), -range / 2
            )  # lower integration bound
            b = min(
                torch.max(data[:chunksize]).item(), range / 2
            )  # upper integration bound
        else:
            a = max(torch.min(data).item(), -range / 2)  # lower integration bound
            b = min(torch.max(data).item(), range / 2)  # upper integration bound
        x_tics = torch.Tensor(np.linspace(a, b, steps)).to(data.device)
    else:
        assert isinstance(x_tics, torch.Tensor)
        x_tics = x_tics.to(data.device)
    x_tics.requires_grad = True
    norm = Normal(loc=target_mean, scale=target_std)
    norm_y_tics = norm.log_prob(x_tics).exp()
    kde_y_tics = kde_prob(
        data, x_tics, bandwidth_scale=bandwidth_scale, depth_limit=depth_limit
    )
    overlap_fn = torch.amin(torch.stack((kde_y_tics, norm_y_tics)), 0)
    overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
    gabe_distance = 1 - overlap_area
    return gabe_distance


def depthwise_gabian_pdf(
    data, x_tics=None, target_mean=0, target_std=1, bandwidth_scale=1, depth_limit=True
):
    """estimates the Depthwise Gabian Probability Density Function of a batch of 1d vectors"""
    chunksize = 64

    def kde_prob(
        data,
        x,
        kernel=Normal(loc=0, scale=target_std),
        bandwidth_scale=1,
        chunksize=chunksize,
        depth_limit=True,
    ):
        """This function is memory intensive"""
        if depth_limit:
            z = data[:chunksize]
        else:
            z = data
        silvermans_factor = ((4 * torch.std(z).pow(5)) / (3 * z.numel())).pow(1 / 5)
        bw = silvermans_factor * bandwidth_scale
        try:
            a = (z.unsqueeze(1) - x.unsqueeze(1)) / bw
            a = kernel.log_prob(a)
            a = torch.exp(a)
            a = bw ** (-1) * a
            a = a.sum(dim=2)
            prob = a / z.shape[1]
        except Exception as e:
            print(x)
            raise e
        return prob

    # ^ this must be an integer
    if x_tics is None:
        steps = 256  # steps for estimation
        range = 9
        if depth_limit:
            a = max(
                torch.min(data[:chunksize]).item(), -range / 2
            )  # lower integration bound
            b = min(
                torch.max(data[:chunksize]).item(), range / 2
            )  # upper integration bound
        else:
            a = max(torch.min(data).item(), -range / 2)  # lower integration bound
            b = min(torch.max(data).item(), range / 2)  # upper integration bound
        x_tics = torch.Tensor(np.linspace(a, b, steps)).to(data.device)
    else:
        assert isinstance(x_tics, torch.Tensor)
        x_tics = x_tics.to(data.device)
    x_tics.requires_grad = True
    norm = Normal(loc=target_mean, scale=target_std)
    norm_y_tics = norm.log_prob(x_tics).exp()
    kde_y_tics = kde_prob(
        data, x_tics, bandwidth_scale=bandwidth_scale, depth_limit=depth_limit
    )
    return kde_y_tics


def depthwise_gabian_distance(
    data, x_tics=None, target_mean=0, target_std=1, bandwidth_scale=1, depth_limit=True
):
    """finds the Depthwise Gabian Distance between a batche of 1d vectors and a batch of standard 
    gaussian distributions (mean=0 and sigma=1)"""
    chunksize = 64

    def kde_prob(
        data,
        x,
        kernel=Normal(loc=0, scale=target_std),
        bandwidth_scale=1,
        chunksize=chunksize,
        depth_limit=True,
    ):
        """This function is memory intensive"""
        if depth_limit:
            z = data[:chunksize]
        else:
            z = data
        # smallest_std = torch.std(z, 0).min()
        # silvermans_factor = ((4*smallest_std.pow(5))/(3*z.numel())).pow(1/5)
        silvermans_factor = ((4 * torch.std(z).pow(5)) / (3 * z.numel())).pow(1 / 5)
        bw = silvermans_factor * bandwidth_scale
        try:
            a = (z.unsqueeze(1) - x.unsqueeze(1)) / bw
            a = kernel.log_prob(a)
            a = torch.exp(a)
            a = bw ** (-1) * a
            a = a.sum(dim=2)
            prob = a / z.shape[1]
        except Exception as e:
            print(x)
            raise e
        return prob

    if x_tics is None:
        steps = 256  # steps for estimation
        range = 9
        if depth_limit:
            a = max(
                torch.min(data[:chunksize]).item(), -range / 2
            )  # lower integration bound
            b = min(
                torch.max(data[:chunksize]).item(), range / 2
            )  # upper integration bound
        else:
            a = max(torch.min(data).item(), -range / 2)  # lower integration bound
            b = min(torch.max(data).item(), range / 2)  # upper integration bound
        x_tics = torch.Tensor(np.linspace(a, b, steps)).to(data.device)
    else:
        assert isinstance(x_tics, torch.Tensor)
        x_tics = x_tics.to(data.device)
    x_tics.requires_grad = True
    norm = Normal(loc=target_mean, scale=target_std)
    norm_y_tics = norm.log_prob(x_tics).exp()
    kde_y_tics = kde_prob(
        data, x_tics, bandwidth_scale=bandwidth_scale, depth_limit=depth_limit
    )
    overlap_fn = torch.amin(
        torch.stack((kde_y_tics, norm_y_tics.repeat(kde_y_tics.shape[0], 1))), 0
    )
    overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
    gabe_distance = torch.mean(1 - overlap_area)
    return gabe_distance


# %matplotlib inline
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# import numpy as np
# fig = plt.figure()
# ax = plt.axes()

# batches = 512
# n = 2055
# samplesN = torch.normal(torch.zeros([batches, n]), torch.ones([batches, n])*1/4).to(device)
# samplesA = torch.normal(-torch.ones([batches, n])*1, torch.ones([n])*1/4).to(device)
# samplesB = torch.normal(torch.ones([batches, n])*1, torch.ones([batches, n])*0.1/4).to(device)
# samplesC = torch.normal(torch.zeros([batches, n]), torch.ones([batches, n])*0.25/4).to(device)
# samples = torch.cat((samplesN, samplesA, samplesB, samplesC), 1).to(device)

# steps = 256  # steps for estimation
# # ^ this must be an integer
# a = -1.1  # lower integration bound
# b = 1.1  # upper integration bound
# x_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)

# bw = 1/2

# normN_y_tics = Normal(loc=0, scale=1/4).log_prob(x_tics).exp().cpu().detach()
# normA_y_tics = Normal(loc=-1, scale=1/4).log_prob(x_tics).exp().cpu().detach()
# normB_y_tics = Normal(loc=1, scale=0.1/4).log_prob(x_tics).exp().cpu().detach()
# normC_y_tics = Normal(loc=0, scale=0.25/4).log_prob(x_tics).exp().cpu().detach()
# normT_y_tics = normN_y_tics + normA_y_tics + normB_y_tics + normC_y_tics

# data = samplesN
# y_tics = gabian_pdf(data, x_tics=x_tics, bandwidth_scale=bw)
# kde_area = torch.trapz(y=y_tics, x=x_tics)
# print("N kde_area:", kde_area)
# ax.plot(x_tics.cpu().detach().numpy(), y_tics.cpu().detach().numpy())
# ax.plot(x_tics.cpu().detach().numpy(), normN_y_tics)
# plt.show()
# overlap_fn = torch.amin(torch.stack((y_tics.to(device), normN_y_tics.to(device))), 0)
# overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
# gabe_distance = (1 - overlap_area)
# print("depth_gabe_distance N:", gabe_distance)
# print("dgd samplesN:", depthwise_gabian_distance(samplesN, target_std=1/4))
# print()

# data = samplesA
# y_tics = gabian_pdf(data, x_tics=x_tics, bandwidth_scale=bw)
# kde_area = torch.trapz(y=y_tics, x=x_tics)
# print("A kde_area:", kde_area)
# # ax.plot(x_tics.cpu().detach().numpy(), y_tics.cpu().detach().numpy())
# ax.plot(x_tics.cpu().detach().numpy(), normA_y_tics)
# plt.show()
# overlap_fn = torch.amin(torch.stack((y_tics.to(device), normA_y_tics.to(device))), 0)
# overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
# gabe_distance = (1 - overlap_area)
# print("depth_gabe_distance A:", gabe_distance)
# print("dgd samplesA:", depthwise_gabian_distance(samplesA, target_std=1/4))
# print()

# data = samplesB
# y_tics = gabian_pdf(data, x_tics=x_tics, bandwidth_scale=bw)
# kde_area = torch.trapz(y=y_tics, x=x_tics)
# print("B kde_area:", kde_area)
# ax.plot(x_tics.cpu().detach().numpy(), y_tics.cpu().detach().numpy())
# ax.plot(x_tics.cpu().detach().numpy(), normB_y_tics)
# plt.show()
# overlap_fn = torch.amin(torch.stack((y_tics.to(device), normB_y_tics.to(device))), 0)
# overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
# gabe_distance = (1 - overlap_area)
# print("depth_gabe_distance B:", gabe_distance)
# print("dgd samplesB:", depthwise_gabian_distance(samplesB, target_std=1/4))
# print()

# data = samplesC
# y_tics = gabian_pdf(data, x_tics=x_tics, bandwidth_scale=bw)
# kde_area = torch.trapz(y=y_tics, x=x_tics)
# print("B kde_area:", kde_area)
# ax.plot(x_tics.cpu().detach().numpy(), y_tics.cpu().detach().numpy())
# ax.plot(x_tics.cpu().detach().numpy(), normC_y_tics)
# plt.show()
# overlap_fn = torch.amin(torch.stack((y_tics.to(device), normC_y_tics.to(device))), 0)
# overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
# gabe_distance = (1 - overlap_area)
# print("depth_gabe_distance C:", gabe_distance)
# print("dgd samplesC:", depthwise_gabian_distance(samplesC, target_std=1/4))
# print()

# data = samples
# y_tics = gabian_pdf(data, x_tics=x_tics, bandwidth_scale=bw)
# kde_area = torch.trapz(y=y_tics, x=x_tics)
# print("T kde_area:", kde_area)
# ax.plot(x_tics.cpu().detach().numpy(), y_tics.cpu().detach().numpy())
# ax.plot(x_tics.cpu().detach().numpy(), normT_y_tics)
# plt.show()
# overlap_fn = torch.amin(torch.stack((y_tics.to(device), normT_y_tics.to(device))), 0)
# overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
# gabe_distance = (1 - overlap_area)
# print("depth_gabe_distance T:", gabe_distance)
# print("dgd samples:", depthwise_gabian_distance(samples, target_std=1/4))


# del samplesN, samplesA, samplesB, samplesC, samples
# torch.cuda.empty_cache()

### Metalog: ###
def ECDF(x: torch.Tensor, dim: int = 0, reach_limits=True):
    """
    set "reach_limit" to false to calculate ECDF in a way that will not include perfect 0 or 1.
    """
    x = torch.sort(x.flatten(dim), dim=dim).values
    n = x.shape[-1]
    cum = torch.arange(1, n + 1).to(x.device) / (n + 1 - reach_limits)
    cum = cum.repeat(*x.shape[0:-1], 1)  # one for each univariate sample
    return x, cum


class Unbounded_Metalog_Model(nn.Module):
    """
    An implimentation of unbounded metalog models.
    """

    def __init__(self, init_a: torch.Tensor = None):
        super(Unbounded_Metalog_Model, self).__init__()

        self.a = nn.Parameter(init_a, requires_grad=True)
        self.n = self.a.shape[-1]

        ### Define basis functions for QF (quantile function):
        def qg1(y, i):
            """first basis function"""
            return torch.ones_like(y)

        def qg2(y, i):
            """second basis function"""
            return torch.log(y / (1 - y))

        def qg3(y, i):
            """third basis function"""
            return (y - 0.5) * torch.log(y / (1 - y))

        def qg4(y, i):
            """fourth basis function"""
            return y - 0.5

        def qgj_odd(y, j):
            """nth odd basis function (after third)"""
            j += 1
            assert (j % 2 != 0) and (j >= 5)
            return (y - 0.5).pow((j - 1) / 2)

        def qgj_even(y, j):
            """nth even basis function (after fourth)"""
            j += 1
            assert (j % 2 == 0) and (j >= 6)
            return torch.log(y / (1 - y)) * (y - 0.5).pow(j / 2 - 1)

        # Start QF basis functions:
        self.qf_basis_functions = [qg1, qg2, qg3, qg4]
        # Additional inverse cdf basis functions as needed:
        self.qf_basis_functions = self.qf_basis_functions + [
            qgj_even if x % 2 == 0 else qgj_odd for x in range(1, self.n - 4 + 1)
        ]
        # Trim as needed:
        self.qf_basis_functions = self.qf_basis_functions[: self.n]

        ### Define basis functions for derivative of quantile function in terms of cumulative
        ### probability. (^ derivative of quantile function):
        def dqg1(y, i):
            """first basis function"""
            return torch.zeros_like(y)

        def dqg2(y, i):
            """second basis function"""
            return 1 / (y * (1 - y))

        def dqg3(y, i):
            """third basis function"""
            return (y - 1 / 2) / (y * (1 - y)) + torch.log(y / (1 - y))

        def dqg4(y, i):
            """fourth basis function"""
            return torch.ones_like(y)

        def dqgj_odd(y, j):
            """nth odd basis function (after third)"""
            j += 1
            assert (j % 2 != 0) and (j >= 5)
            return ((j - 1) / 2) * (y - 1 / 2).pow((j - 3) / 2)

        def dqgj_even(y, j):
            """nth even basis function (after fourth)"""
            j += 1
            assert (j % 2 == 0) and (j >= 6)
            return (y - 1 / 2).pow(j / 2 - 1) / (y * (1 - y)) + (j / 2 - 1) * (
                y - 1 / 2
            ).pow(j / 2 - 2) * torch.log(y / (1 - y))

        # Start derivative quantile basis functions:
        self.dqf_basis_functions = [dqg1, dqg2, dqg3, dqg4]
        # Additional dqf basis functions as needed:
        self.dqf_basis_functions = self.dqf_basis_functions + [
            dqgj_odd if x % 2 == 0 else dqgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        self.dqf_basis_functions = self.dqf_basis_functions[: self.n]

    def constrain(self):
        """Coefficients are unconstrained in this case."""
        pass

    def quantile(self, y):
        """
        Quantile of cumulative probability "y".  (returns x-position of cumulative probability "y".
        This is an inverse CDF)
        """
        x_values = sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.qf_basis_functions)
            ]
        )
        return x_values

    def derivative_quantile(self, y):
        """
        Derivative of quantile as function of cumulative probability "y".
        (AKA: quantile density function.)
        """
        return sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.dqf_basis_functions)
            ]
        )

    def prob_ito_cumprob(self, y):
        """Probability density in terms of cumulative probability "y"."""
        return self.derivative_quantile(y).pow(-1)

    def prob(self, x, iters=64):
        """
        Approximates probability density at a batch of tensors "x" by asymptotically bounded
        approach. There is currently no known closed-form inverse metalog.
        """
        eps = 1e-7
        cum_y_guess = torch.ones_like(x) * 1 / 3

        lr = 1 / 3
        old_x_guess = self.quantile(cum_y_guess)  # initial
        old_diff = 0  # initial
        adj = torch.tensor([1]).to(self.a.device) / x.shape[1]  # initial
        for i in range(iters):
            cum_y_guess += adj
            x_guess = self.quantile(cum_y_guess)
            diff = x - x_guess
            #  print(f"mean squared diff {i}:", (torch.sum(diff.pow(2))/diff.shape[1]).item())
            max_adj = (
                torch.heaviside(diff, torch.Tensor([0]).to(cum_y_guess)).clamp(
                    min=eps, max=1 - eps
                )
                - cum_y_guess
            )
            adj = max_adj * torch.tanh(diff.pow(2)) * lr

        density = self.prob_ito_cumprob(cum_y_guess)
        density = torch.nan_to_num(density, nan=0)
        return density

    def log_prob(self, x):
        """Approximates log of probability density at a batch of tensors "x"."""
        return torch.log(self.prob(x))

    def estimate_entropy(self, steps=256):
        """Estimates shannon entropy of the distribution in nats by numeric integration."""
        self.a.data = self.a.data.double()  # increase precision
        eps = 1e-7
        a = eps  # lower integration bound
        b = 1 - eps  # upper integration bound
        cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).to(self.a.device).double()
        # shape for batch and channel support;
        cum_y_tics = cum_y_tics.repeat(self.a.shape[0], 1)

        qp_tics = self.derivative_quantile(cum_y_tics)
        entropy = torch.trapz(torch.nan_to_num(torch.log(qp_tics), 0), cum_y_tics)

        # x_tics = self.quantile(cum_y_tics)
        # p_tics = self.prob_ito_cumprob(cum_y_tics)
        # entropy = -torch.trapz(p_tics*torch.nan_to_num(torch.log(p_tics), 0), x_tics)

        self.a.data = self.a.data.float()  # reset precision

        return entropy

    def sample(self, shape):
        """Simulates data of shape "shape" by inverse tranform sampling."""
        eps = 1e-7
        return self.quantile(
            torch.rand(shape).clamp(min=eps, max=1 - eps).to(self.a.device)
        )

    def forward(self, x):
        """
        By default: Approximates probability density at a batch of tensors "x" by asymptotically
        bounded approach. There is currently no known closed-form inverse metalog.
        """
        return self.prob(x)


def Metalog_Fit_Closed_Form(model, data):
    """
    Fits the parameters of the metalog model, "model", to sources of data in "data", by a closed-form
    linear least-squares method.
    This function supports batching for fitting many datasets at once and expects data in batched
    form (with at least 2 dimensional shape). First dimension of data must match first dimension of
    model coefficients "a". If first dimension > 1 (batchsize > 1), this function will fit a number
    of sets of coefficients, namely: one set of coefficients in the provided metalog model for each
    dataset, where the first-dimension or "batch-size" of "data" indicates the number of independent
    datasets.
    """
    ecdf = ECDF(data, dim=1, reach_limits=False)
    x, y = ecdf
    x = x.float()
    y = y.float()

    Y_cols = [f(y, idx) for idx, f in enumerate(model.qf_basis_functions)]
    Y = torch.stack(Y_cols, -1)
    a = torch.bmm(
        torch.linalg.solve(torch.bmm(Y.transpose(1, 2), Y), Y.transpose(1, 2)),
        x.unsqueeze(-1),
    ).flatten(1)
    model.a.data = a


def polynomial_fit(x, y, num_terms=5, weights=None):
    """
    Fits a polynomial curve of order "num_terms" to the datapoints in "x" and "y". This function
    supports channeling for fitting to mulitple independent data channels at once, and expects data
    in channeled form with shape like [channels, points]. (one channel is fine)
    Returns Tensor of shape [channels, num_terms].
    """
    X_cols = [x.pow(n) for n in range(num_terms)]
    X = torch.stack(X_cols, -1)

    if weights is None:
        a = torch.bmm(
            torch.linalg.solve(torch.bmm(X.transpose(1, 2), X), X.transpose(1, 2)),
            y.unsqueeze(-1),
        ).flatten(1)
    else:
        a = torch.bmm(
            torch.linalg.solve(
                torch.bmm(X.transpose(1, 2), torch.bmm(weights, X)), X.transpose(1, 2)
            ),
            torch.bmm(weights, y.unsqueeze(-1)),
        ).flatten(1)
    return a


def mutual_information(data, idx, num_terms=7, weights=None):
    """
    Returns the mutual information between the channel "idx" of the data, and every channel of the
    data (including channel idx, which should be very large if nothing went wrong).
    In other words:
        this measures how much knowledge is to be gained about other channels from knowledge of the
        state of channel "idx".
    """
    channels = data.shape[0]

    # calculate initial entropy
    # model = Unbounded_Metalog_Model(init_a=torch.zeros([channels, 5])).to(
    #     device
    # )
    # Metalog_Fit_Closed_Form(model, data)
    # initial_entropy = model.estimate_entropy(steps=512)

    # prepare to fit curve to relationship
    x = data[idx].repeat(channels, 1)
    y = data

    # fit curve to relationship
    X_cols = [x.pow(n) for n in range(num_terms)]
    X = torch.stack(X_cols, -1)

    if weights is None:
        a = torch.bmm(
            torch.linalg.solve(torch.bmm(X.transpose(1, 2), X), X.transpose(1, 2)),
            y.unsqueeze(-1),
        ).flatten(1)
    else:
        a = torch.bmm(
            torch.linalg.solve(
                torch.bmm(X.transpose(1, 2), torch.bmm(weights, X)), X.transpose(1, 2)
            ),
            torch.bmm(weights, y.unsqueeze(-1)),
        ).flatten(1)

    # calculate confounding effect
    y = torch.mm(a, X[0, :].T)

    # subtract that effect
    data = data - y

    # calcualte new entropy:
    model = Unbounded_Metalog_Model(init_a=torch.zeros([channels, 10])).to(device)
    Metalog_Fit_Closed_Form(model, data)
    final_entropy = model.estimate_entropy(steps=512)

    return final_entropy


from torchvision.transforms.functional import InterpolationMode

# DATA_ROOT = "/content/drive/MyDrive/LossMultiplexing/data/processedAnimeFaces-512x512/"
DATA_ROOT = "/content/drive/LossMultiplexing/data/processedAnimeFaces-512x512/"
DATA_CHUNKS = os.listdir(DATA_ROOT)
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(128, interpolation=InterpolationMode.BILINEAR),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
ANIME_FACES = []
ANIME_FACES.extend(torch.load(os.path.join(DATA_ROOT, "128x128.chnk")))
# ANIME_FACES.extend(
#             torch.load(
#                 os.path.join(
#                     DATA_ROOT, "256x256.chnk"
#                     )
#                 )
#             )


def fetch_chunk():
    ANIME_FACES.clear()
    while True:
        try:  # there are some files that can't be loaded
            # ^ so just try again with a new one.
            if len(DATA_CHUNKS) > 0:
                chunk_name = np.random.choice(DATA_CHUNKS)
                DATA_CHUNKS.remove(chunk_name)
                ANIME_FACES.extend(
                    [
                        TRANSFORM(x)
                        for x in tqdm(torch.load(os.path.join(DATA_ROOT, chunk_name)))
                    ]
                )
                break
            else:
                print("DATA EXHAUSTED")
                break
        except Exception:
            continue


def refresh_chunks():
    DATA_CHUNKS = os.listdir(DATA_ROOT)


# fetch_chunk()
# torch.save(ANIME_FACES, os.path.join(DATA_ROOT, "128x128.chnk"))

# Hyperparameters etc.
CHANNELS_IMG = 3
IMAGE_WIDTH = 256

# Z_DIM = 3496
# # (6 * 16^2) + (10 * 7^2) + (9 * 7^2) + (8 * 7^2) + (7 * 7^2) + (6 * 7^2)
# Z_DIM = 2750
# # (5 * 16^2) + (9 * 7^2) + (8 * 7^2) + (7 * 7^2) + (6 * 7^2)
# Z_DIM = 2053
# (4 * 16^2) + (8 * 7^2) + (7 * 7^2) + (6 * 7^2)
# Z_DIM = 3376
# # (4 * 16^2) + (16 * 7^2 * 3)
# Z_DIM = 3136
# # (16 * 7^2 * 4)
# Z_DIM = 5488
# (16 * 7^2 * 7)
# Z_DIM = 4096
# # (8^4)
Z_DIM = 512
# (8^2 * 8)

NUM_CODE_CRITICS = 0
NUM_ICRITICS = 0
FEATURES_ENC = 64
FEATURES_DEC = 64
FEATURES_ICRITIC = 16
FEATURES_CODE_CRIT = "N/A"

DAE_LEARNING_RATE = 1 / 2
β = 1  # disentanglement factor

ICRITIC_LEARNING_RATE = 5e-3

CODE_CRITIC_LEARNING_RATE = 1e-3
ICRITIC_ITERATIONS = 1
CODE_CRITIC_ITERATIONS = 1
LAMBDA_GP = 10  # gradient penalty factor

BATCH_SIZE = 128
BATCH_MULTIPLIER = 1
# TOTAL_BATCH_SIZE = BATCH_SIZE * BATCH_MULTIPLIER * CODE_CRITIC_ITERATIONS
TOTAL_BATCH_SIZE = BATCH_SIZE * BATCH_MULTIPLIER
NUM_EPOCHS = 5

kwargs = {"num_workers": 4, "pin_memory": True}

dataset = AnimeFacesDataset()

loader = torch.utils.data.DataLoader(
    dataset=dataset,
    sampler=RandomSampler(dataset),
    batch_size=TOTAL_BATCH_SIZE,
    drop_last=True,
    **kwargs,
)


def refresh_chunks():
    DATA_CHUNKS = os.listdir(DATA_ROOT)


def fetch_chunk():
    ANIME_FACES.clear()
    while True:
        try:  # there are some files that can't be loaded
            # ^ so just try again with a new one.
            if len(DATA_CHUNKS) > 0:
                chunk_name = np.random.choice(DATA_CHUNKS)
                DATA_CHUNKS.remove(chunk_name)
                ANIME_FACES.extend(
                    [
                        TRANSFORM(x)
                        for x in tqdm(torch.load(os.path.join(DATA_ROOT, chunk_name)))
                    ]
                )
                break
            else:
                print("DATA EXHAUSTED")
                break
        except Exception:
            continue


def refresh_data():
    fetch_chunk()
    dataset = AnimeFacesDataset()
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=RandomSampler(dataset),
        batch_size=TOTAL_BATCH_SIZE,
        drop_last=True,
        **kwargs,
    )


# refresh_data()


def initialize_dae(model):
    for m in model.modules():
        # print("\nm:", m)
        try:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        except:
            continue
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.constant_(m.weight, -1 / fan_in)
            m.weight.data.flatten()[
                torch.multinomial(
                    torch.ones_like(m.weight.flatten()), m.weight.numel() // 2
                )
            ] = (1 / fan_in)
            nn.init.constant_(m.bias, 0)
        elif isinstance(
            m,
            (
                nn.BatchNorm2d,
                nn.BatchNorm1d,
                nn.LayerNorm,
                nn.InstanceNorm2d,
                nn.InstanceNorm1d,
            ),
        ):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# load:
# dae = torch.load("models/DAE_dae_UNITARY_BASE_animefaces.ckpt")
# dae = torch.load("models/DAE_dae_1_animefaces.ckpt")
# dae = torch.load("models/DAE_dae_1_TEST2_animefaces.ckpt")
# dae = torch.load("models/DAE_dae_First_Success_Adam_equ_wght_animefaces.ckpt")
# dae = torch.load("models/DAE_dae_1_TEST_BC_animefaces.ckpt")
# dae = torch.load("models/DAE_dae_1_animefaces.ckpt")
# dae = torch.load("models/DAE_dae_256_1_animefaces.ckpt").to(device)
# dae = torch.load("models/DAE_dae_128_1_animefaces.ckpt").to(device)
# dae = torch.load("models/DAE_dae_tmi_1_animefaces.ckpt").to(device)
# init:
dae = DAE(Z_DIM, CHANNELS_IMG, FEATURES_ENC, FEATURES_DEC).to(device)
# initialize_dae(dae)
# initialize_clear(dae)
initialize_kaiming_he(dae, a=0.2, nonlinearity="leaky_relu")
# initialize_kaiming_he_big(dae, c=10, a=0.2, nonlinearity="leaky_relu")

# set initialization statistics on a large batch
# with torch.no_grad():
#     target_std = 1
#     temp_batch_size = 256
#     img_noise = torch.normal(
#         torch.zeros([temp_batch_size, CHANNELS_IMG, IMAGE_WIDTH, IMAGE_WIDTH]),
#         torch.ones([temp_batch_size, CHANNELS_IMG, IMAGE_WIDTH, IMAGE_WIDTH]) * target_std,
#     ).to(device).clamp(min=-4.25 * target_std, max=4.25 * target_std)
#     dae(img_noise)
#     dae.zero_grad()
#     del img_noise

with torch.no_grad():
    temp_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=RandomSampler(dataset),
        batch_size=512,
        drop_last=True,
        **kwargs,
    )

    dae(next(iter(temp_loader)).to(device))
    dae.zero_grad()
    del temp_loader

stack_height = 8
# load:
# icritic = torch.load("models/DAE_icritic_1_animefaces.ckpt")
# icritic = torch.load("models/DAE_icritic_1_TEST2_animefaces.ckpt")
# icritic = torch.load("models/DAE_icritic_1_TEST_BC_animefaces.ckpt")
# icritics = [torch.load(f"models/DAE_icritic{idx}_1_animefaces.ckpt").to(device) for idx in range(NUM_ICRITICS)]
# init:
# icritic = Critic(CHANNELS_IMG, FEATURES_ICRITIC).to(device)
# initialize_kaiming_he(c, a=0.2, nonlinearity="leaky_relu")
# icritics = [Critic(CHANNELS_IMG*stack_height, FEATURES_ICRITIC).to(device) for _ in range(NUM_ICRITICS)]
# for c in icritics:
#     initialize_kaiming_he(c, a=0.2, nonlinearity="leaky_relu")

# load:
# code_critics = [torch.load(f"models/DAE_code_critic{x}_FINAL_BASE_TEST_animefaces.ckpt") for x in range(NUM_CODE_CRITICS)]
# init:
# code_critics = [CodeCritic(Z_DIM).to(device) for _ in range(NUM_CODE_CRITICS)]
# for c in code_critics:
#     initialize_kaiming_he(c, nonlinearity="leaky_relu")

# summary(code_critics[0], (Z_DIM))

# for c in code_critics:
#     c.train()

vgg = PerceptualLoss(model=torchvision.models.vgg16, depth=1).to(device).eval()
effnet = (
    PerceptualLoss(model=torchvision.models.efficientnet_b0, depth=1).to(device).eval()
)
# effnet_model = PerceptualLoss(model=torchvision.models.efficientnet_b0, depth=2).to(device).eval()
# def effnet(x, y):
#     return effnet_model(x, y)[1]
mobilenet_v3 = (
    PerceptualLoss(model=torchvision.models.mobilenet_v3_small, depth=1)
    .to(device)
    .eval()
)
densenet = (
    PerceptualLoss(model=torchvision.models.densenet161, depth=1).to(device).eval()
)
squeezenet = (
    PerceptualLoss(model=torchvision.models.squeezenet1_0, depth=1).to(device).eval()
)

dae_lm = Loss_Multiplex(
    goal_functions=[
        bce_loss,  # binary cross entropy loss
        vgg,  # first and second VGG perceptual loss NOTE: Returns multiple losses
        effnet,  # first and second EfficientNet perceptual loss NOTE: Returns multiple losses
        mobilenet_v3,  # another perceptual loss
        densenet,  # another perceptual loss
        squeezenet,  # another perceptual loss
        # lambda x: torch.mean(x), # WGAN loss
    ],
    # scale_factors=[
    #     # 1/0.025,  # binary cross entropy scale
    #     # 1/0.05,  # first VGG scale
    #     # 1/0.25,  # second VGG scale
    #     # 1/3,  # first EfficientNet scale
    #     # 1/0.03,  # second EfficientNet scale
    #     # # 1/10,  # WGAN scale
    #     # 1.45,  # binary cross entropy scale
    #     # 0.589,  # first VGG scale
    #     # 0.164,  # second VGG scale
    #     # 0.0130,  # first EfficientNet scale
    #     # 1.625,  # second EfficientNet scale
    #     # # 1/180,  # WGAN scale
    #     # 15.3,  #  bce
    #     # 0.281,  # vgg1
    #     # 0.0848,  # vgg2
    #     # 0.0255,  # effnet1
    #     # 0.734,  # effnet2
    #     # 0.260,  # mobilenet1
    #     # 1.304,  # mobilenet2
    #     # 42.8,  # densenet1
    #     # 292,  # densenet2
    #     # 0.0653,  # squeezenet1
    #     # 0.00292,  # squeezenet2
    # ],
    scale_factors="auto",
    num_losses=6,
    base_window=2,
    window_step=1,
    loss_weight_pursuit_rate=1 / 2,
    auto_pursuit=False,
    pool_size=BATCH_MULTIPLIER,
)

# ic_lm = Loss_Multiplex(
#     goal_functions=[
#         lambda x: torch.mean(x) for _ in range(NUM_ICRITICS)
#     ],
#     scale_factors=[1 for _ in range(NUM_ICRITICS)],
#     num_losses=NUM_ICRITICS,
#     base_window=2,
#     window_step=1,
#     loss_weight_pursuit_rate=1/2,
#     auto_pursuit=False,
#     pool_size=BATCH_MULTIPLIER,
# )

# cc_lm = Loss_Multiplex(
#     goal_functions=[
#         lambda x: torch.mean(x) for _ in range(NUM_CODE_CRITICS)
#     ],
#     scale_factors=[1 for _ in range(NUM_CODE_CRITICS)],
#     num_losses=NUM_CODE_CRITICS,
#     base_window=2,
#     window_step=1,
#     loss_weight_pursuit_rate=1,
#     auto_pursuit=False,
#     pool_size=BATCH_MULTIPLIER,

## initializate vae optimizer
# opt_dae = optim.Adam(dae.parameters(), lr=DAE_LEARNING_RATE)
# opt_dae = optim.Adam(dae.parameters(), lr=DAE_LEARNING_RATE, betas=(0.0, 0.9), weight_decay=0)
# opt_dae = optim.RMSprop(dae.parameters(), lr=DAE_LEARNING_RATE)
# opt_dae = optim.SGD(dae.parameters(), lr=DAE_LEARNING_RATE)
opt_dae = optim.Adadelta(dae.parameters(), lr=DAE_LEARNING_RATE)

# initializate image critic optimizer
# opt_icritics = [optim.Adam(c.parameters(), lr=ICRITIC_LEARNING_RATE, betas=(0.0, 0.9)) for c in icritics]
# opt_icritic = optim.Adam(icritic.parameters(), lr=ICRITIC_LEARNING_RATE, betas=(0.0, 0.9))
# opt_icritic = optim.RMSprop(icritic.parameters(), lr=ICRITIC_LEARNING_RATE)
# opt_icritic = optim.SGD(icritic.parameters(), lr=ICRITIC_LEARNING_RATE)
# opt_icritic = optim.Adadelta(icritic.parameters(), lr=ICRITIC_LEARNING_RATE)

# initializate code critic optimizers
# opt_code_critics = [optim.Adam(c.parameters(), lr=CODE_CRITIC_LEARNING_RATE, betas=(0.0, 0.9)) for c in code_critics]
# opt_code_critics = [optim.RMSprop(c.parameters(), lr=CODE_CRITIC_LEARNING_RATE) for c in code_critics]
# opt_code_critics = [optim.SGD(c.parameters(), lr=CODE_CRITIC_LEARNING_RATE) for c in code_critics]
# opt_code_critics = [optim.Adadelta(c.parameters(), lr=CODE_CRITIC_LEARNING_RATE) for c in code_critics] )

wandb.init(project="HIDAE_Live_AnimeFaces128", entity="examday")
# wandb.init(project="HIDAE_AnimeFaces128", entity="examday", id='devout-vortex-399', resume=True)

wandb.config.update(
    {
        "channels_img": CHANNELS_IMG,
        "img width": IMAGE_WIDTH,
        "Z dim": Z_DIM,
        "num code critics": NUM_CODE_CRITICS,
        "num image critics": NUM_ICRITICS,
        "features encoder": FEATURES_ENC,
        "features decoder": FEATURES_DEC,
        "features image critic": FEATURES_ICRITIC,
        "features code critic": FEATURES_CODE_CRIT,
        "dae learning rate": DAE_LEARNING_RATE,
        "β": β,
        "image critic learning_rate": ICRITIC_LEARNING_RATE,
        "code critic learning_rate": CODE_CRITIC_LEARNING_RATE,
        "image critic iterations": ICRITIC_ITERATIONS,
        "code critic iterations": CODE_CRITIC_ITERATIONS,
        "lambda gp": LAMBDA_GP,
        "batch size": BATCH_SIZE,
        "batch multiplier": BATCH_MULTIPLIER,
        "effective batch size": BATCH_SIZE * BATCH_MULTIPLIER,
        "total batch size": TOTAL_BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "num vae losses": dae_lm.num_losses,
        # "num cc losses": cc_lm.num_losses,
    }
)

from copy import deepcopy
from pprint import pprint

logInterval = 6
target_std = 1  # at this point the density at -4 and 4 is very close to 1/10,000

metalog_model = Unbounded_Metalog_Model(init_a=torch.zeros([512, 10])).to(device)


def generate_image(z):
    with torch.no_grad():
        # gz = torch.normal(
        #     torch.zeros([BATCH_SIZE, Z_DIM]),
        #     torch.ones([BATCH_SIZE, Z_DIM]) * target_std,
        # ).to(device)
        # return dae.decoder(gz)

        # Metalog_Fit_Closed_Form(metalog_model, z.T)
        simulated_z = metalog_model.sample([BATCH_SIZE, Z_DIM, 1]).squeeze(-1)
        return dae.decoder(simulated_z)


# with torch.autograd.detect_anomaly():
first = True
np.set_printoptions(formatter={"float": "{: 0.5f}".format})
# tot_old_z = torch.zeros([BATCH_MULTIPLIER, BATCH_SIZE, Z_DIM]).to(device)
# old_code_disc_progresses = [torch.Tensor([0]).to(device) for _ in range(NUM_CODE_CRITICS)]
old_combo_disc_progs = [torch.Tensor([0]).to(device) for _ in range(NUM_ICRITICS)]

sh = stack_height  # stack height for critic
cbs = 8  # critic batch size

step = 0
max_steps = 1000000
for epoch in range(1, NUM_EPOCHS + 1):
    with tqdm(total=max_steps) as pbar:
        # with tqdm(total=len(loader)) as pbar:
        chunk = 0
        Still_Chunks = True
        pbar.set_description(f"{epoch}, {chunk}")
        while Still_Chunks and step < max_steps:
            for tot_batch in loader:
                start = 0
                wlog = {}

                ##################
                ### train DAE: ###
                dae.train()
                for i in range(BATCH_MULTIPLIER):
                    real = tot_batch[start : start + BATCH_SIZE].to(device)
                    lab_real = rgb_to_lab_batch(real)
                    start += BATCH_SIZE

                    # noiseyTime = 5000
                    # if step < noiseyTime:
                    #     img_noise = torch.normal(
                    #         torch.zeros([BATCH_SIZE, CHANNELS_IMG, IMAGE_WIDTH, IMAGE_WIDTH]),
                    #         torch.ones([BATCH_SIZE, CHANNELS_IMG, IMAGE_WIDTH, IMAGE_WIDTH]) * target_std,
                    #     ).to(device).clamp(min=-4.25 * target_std, max=4.25 * target_std)
                    #     # add noise to image and decay noise over time
                    #     sn_ratio = step/noiseyTime
                    #     real = sn_ratio*(real-0.5) + (1-sn_ratio)*img_noise
                    #     # normalize to fit within zero and one:
                    #     # real -= torch.min(real)
                    #     # real /= torch.max(real)

                    recon, z, uncut_z = dae(real, include_uncut=True)
                    lab_recon = rgb_to_lab_batch(recon)

                    #################
                    ### dae multiloss: ###
                    dae_multiloss, dae_losses, dae_scaled_losses, dae_weighted_losses, dae_weights = dae_lm.loss(
                        (lab_recon, lab_real),
                        (recon, real),
                        (recon, real),
                        (recon, real),
                        (recon, real),
                        (recon, real),
                        # (ic_multiloss + 1) / 256,
                        # recon_disc_progress,
                        step_weights=i == BATCH_MULTIPLIER - 1,
                    )
                    #################

                    ##########################
                    ### gen critic loss: ###
                    # lab_real = lab_real.reshape(BATCH_SIZE//sh, lab_real.shape[1]*sh, lab_real.shape[2], lab_real.shape[3])
                    # # ^ Stack multiple instances.

                    # # noise = torch.normal(
                    # #     torch.zeros([BATCH_SIZE, Z_DIM]),
                    # #     torch.ones([BATCH_SIZE, Z_DIM]) * target_std,
                    # # ).to(device)
                    # # gen = dae.decoder(noise)
                    # # lab_gen = rgb_to_lab_batch(gen)
                    # # lab_gen = lab_gen.reshape(BATCH_SIZE//sh, lab_gen.shape[1]*sh, lab_gen.shape[2], lab_gen.shape[3])
                    # # ^ Stack multiple instances.
                    # lab_recon = lab_recon.reshape(BATCH_SIZE//sh, lab_recon.shape[1]*sh, lab_recon.shape[2], lab_recon.shape[3])
                    # # ^ Stack multiple instances.

                    # target_icrits = [torch.mean(c(lab_real)) for c in icritics]
                    # combo = lab_recon
                    # # combo = torch.cat((lab_gen, lab_recon), 0)

                    # now_icrits = [torch.mean(c(combo)) for c in icritics]
                    # now_combo_discs = [x - y for x, y in zip(target_icrits, now_icrits)]

                    # if first:
                    #     old_combo = combo.detach().to("cpu")
                    #     first = False

                    # # to reduce Bar Confusion:
                    # with torch.no_grad():
                    #     last_icrits = [torch.mean(c(old_combo.to(device))) for c in icritics]
                    #     last_combo_discs = [x - y for x, y in zip(target_icrits, last_icrits)]

                    # old_combo = combo.detach().to("cpu")
                    # delta_combo_discs = [x - y for x, y in zip(now_combo_discs, last_combo_discs)]
                    # eps = 1e-16
                    # for idx, delta in enumerate(delta_combo_discs):
                    #     if delta == 0:
                    #         delta_combo_discs[idx] = np.random.choice([-1, 1]) * eps
                    # combo_disc_progresses = [x + y for x, y in zip(old_combo_disc_progs, delta_combo_discs)]
                    # old_combo_disc_progs = [x.detach() for x in combo_disc_progresses]
                    # ic_multiloss, ic_losses, ic_scaled_losses, ic_weighted_losses, ic_weights = ic_lm.loss(
                    #         *combo_disc_progresses,
                    #         step_weights = i == BATCH_MULTIPLIER - 1,
                    #     )
                    ##########################

                    ######################################
                    ### code critic multiloss: ###
                    # for c in code_critics:
                    #     c.eval()

                    # if first:
                    #     tot_old_z[i, :, :] = z.detach().to("cpu")
                    #     first = False

                    # noise = torch.normal(
                    #     torch.zeros([BATCH_SIZE, Z_DIM]),
                    #     torch.ones([BATCH_SIZE, Z_DIM]) * target_std,
                    # ).to(device)

                    # target_code_crits = [torch.mean(c(noise)) for c in code_critics]
                    # now_code_crits = [torch.mean(c(z)) for c in code_critics]
                    # now_code_discs = [x - y for x, y in zip(target_code_crits, now_code_crits)]

                    # # to reduce Bar Confusion:
                    # old_z = tot_old_z[i]
                    # with torch.no_grad():
                    #     # finding mean of performance opinions across multiple batches.
                    #     last_code_crits = [torch.mean(c(old_z)) for c in code_critics]
                    #     last_code_discs = [x - y for x, y in zip(target_code_crits, last_code_crits)]
                    # tot_old_z[i, :, :] = z.detach().to("cpu")

                    # delta_code_discs = [x - y for x, y in zip(now_code_discs, last_code_discs)]

                    # eps = 1e-16
                    # for idx, dc in enumerate(delta_code_discs):
                    #     if dc == 0:
                    #         delta_code_discs[idx] += np.random.choice([-1, 1]) * eps

                    # code_disc_progresses = [x + y for x, y in zip(old_code_disc_progresses, delta_code_discs)]
                    # old_code_disc_progresses = [c.detach() for c in code_disc_progresses]
                    # cc_multiloss, cc_losses, cc_scaled_losses, cc_weighted_losses, cc_weights = cc_lm.loss(*code_disc_progresses, step_weights = i == BATCH_MULTIPLIER - 1)
                    # scaled_cc_loss = cc_multiloss * 10  # times code critic scale factor
                    ######################################

                    # depth_gabe_loss = depthwise_gabian_distance(uncut_z, target_std=target_std, bandwidth_scale=1, depth_limit=False)
                    # scaled_depth_gabe_loss = depth_gabe_loss * 75

                    # out_of_range_penalty = F.threshold(uncut_z.abs(), 4.5*target_std, 0).sum()/uncut_z.numel()

                    # nsl = node_stat_loss(uncut_z, target_std=target_std)
                    # scaled_nsl = nsl * 0.5

                    # entanglement = β*(scaled_depth_gabe_loss + out_of_range_penalty + scaled_nsl)

                    # mode_loss = modularity_loss(uncut_z)
                    # scaled_mode_loss = 1.5785 * mode_loss  # makes loss pass through (0.555, 0.555)

                    # std = torch.std(uncut_z, 0)
                    # std_loss = (std - target_std).pow(2).mean()

                    Metalog_Fit_Closed_Form(metalog_model, z.T)
                    entropy = metalog_model.estimate_entropy(steps=1024)

                    tmi = 0
                    for i in range(Z_DIM):
                        final_ent = mutual_information(z.T[i:], 0)
                        mi = -(final_ent - entropy[i:])
                        mi[0] = 0
                        tmi += torch.sum(mi) / ((Z_DIM + 1) * Z_DIM / 2)

                    entropy = entropy.mean()

                    # dae_loss = (dae_multiloss + entanglement) / BATCH_MULTIPLIER
                    # dae_loss = (dae_multiloss + (gen_disc_progress / 720)) / BATCH_MULTIPLIER
                    # dae_loss = (dae_multiloss + scaled_mode_loss)
                    # dae_loss = dae_multiloss + (std_loss * 1) - (entropy * 1)
                    dae_loss = (
                        dae_multiloss
                        - entropy.pow(2) * torch.sign(entropy) * 10
                        + tmi * 100
                    )
                    dae_loss.backward()

                ##################
                ### log stuff: ###
                if step % logInterval == 0:
                    with torch.no_grad():
                        # record DAE examples:
                        dae_image_array = torchvision.utils.make_grid(
                            torch.cat((real, recon), 0), nrow=BATCH_SIZE
                        )
                        dae_images = wandb.Image(dae_image_array, caption="")

                        # make generation examples:
                        generated_image = generate_image(z)
                        # generated_image = gen
                        generated_image_array = torchvision.utils.make_grid(
                            generated_image, nrow=BATCH_SIZE
                        )
                        generated_image = wandb.Image(generated_image_array, caption="")

                        # update the wandb log:
                        wlog.update(
                            {
                                "DAE examples": dae_images,
                                "generation examples": generated_image,
                            }
                        )
                ##################

                opt_dae.step()
                opt_dae.zero_grad()
                dae.eval()
                ##################

                ###########################
                ### train image critic: ###
                # for _ in range(ICRITIC_ITERATIONS):
                #     for i in range(BATCH_SIZE//sh//cbs):
                #         ins_real = lab_real[i*cbs:(i+1)*cbs].detach()
                #         ins_real.requires_grad = True

                #         # ins_gen = lab_gen[i*cbs:(i+1)*cbs].detach()
                #         ins_recon = lab_recon[i*cbs:(i+1)*cbs].detach()

                #         ins_combo = ins_recon
                #         ins_combo.requires_grad = True

                #         # ins_combo = torch.cat((ins_gen[:ins_gen.shape[0]//2], ins_recon[:ins_recon.shape[0]//2]), 0)
                #         # ins_combo = combo.detach()
                #         # ins_combo.requires_grad = True

                #         image_discernments = []
                #         for c, opt_c in zip(icritics, opt_icritics):
                #             c.train()
                #             icritic_real = c(ins_real).flatten(0)
                #             icritic_gen = c(ins_combo).flatten(0)
                #             # to calculate gradient penalty:
                #             igp = image_gradient_penalty(c, ins_real, ins_combo, device=device)
                #             image_discernment = torch.mean(icritic_real) - torch.mean(icritic_gen)
                #             image_discernments.append(image_discernment.detach())
                #             loss_icritic = (
                #                 -(image_discernment) + LAMBDA_GP * igp
                #             )/BATCH_MULTIPLIER

                #             loss_icritic.backward()
                #             # nn.utils.clip_grad_norm_(icritic.parameters(), max_norm=2.0, norm_type=2)

                #             opt_c.step()
                #             opt_c.zero_grad()
                #             c.eval()
                ###########################

                # clean memory
                # del ins_real, ins_combo, real, lab_real, recon, lab_recon
                # torch.cuda.empty_cache()

                #########################
                ### train code critic: ###
                # to train critic on same sample every time:
                # for critic in code_critics:
                #     critic.train()

                # for i in range(CODE_CRITIC_ITERATIONS):
                #     ## Code Critic Train: ##
                #     cbs = 6
                #     cstart = 0
                #     for j in range(BATCH_SIZE//cbs):
                #         code = z[cstart:cstart + cbs]
                #         cstart += cbs
                #         noise = torch.normal(
                #             torch.zeros([cbs, Z_DIM]),
                #             torch.ones([cbs, Z_DIM]),
                #         ).to(device)
                #         # step
                #         code_discernments = [0 for _ in range(NUM_CODE_CRITICS)]
                #         for k, code_critic in enumerate(code_critics):
                #             ins_noise = noise.detach()
                #             ins_noise.requires_grad = True

                #             ins_z = code.detach()
                #             ins_z.requires_grad = True

                #             code_critic_real = code_critic(ins_noise).reshape(-1)
                #             code_critic_gen = code_critic(ins_z).reshape(-1)
                #             # to calculate gradient penalty:
                #             gp = gradient_penalty(code_critic, ins_noise, ins_z, device=device)
                #             code_discernment = torch.mean(code_critic_real) - torch.mean(code_critic_gen)
                #             code_discernments[k] += code_discernment / BATCH_MULTIPLIER
                #             loss_code_critic = (
                #                 -(code_discernment) + LAMBDA_GP * gp
                #             )/BATCH_MULTIPLIER
                #             code_critic.zero_grad()
                #             loss_code_critic.backward()
                #             # nn.utils.clip_grad_value_(dae.parameters(), clip_value=1.0)
                #             nn.utils.clip_grad_norm_(code_critic.parameters(), max_norm=2.0, norm_type=2)

                #         for opt_code_critic in opt_code_critics:
                #             opt_code_critic.step()
                #             opt_code_critic.zero_grad()

                # for code_critic in code_critics:
                #     code_critic.eval()
                #########################

                ##################
                ### log stuff: ###
                dae_lm_names = [
                    "bce",
                    "vgg1",
                    # "vgg2",
                    "effnet1",
                    # "effnet2",
                    "mobilenet_v3_1",
                    # "mobilenet_v3_2",
                    "densenet1",
                    # "densenet2",
                    "squeezenet1",
                    # "squeezenet2",
                    "icritic",
                ]

                node_std, node_mean = torch.std_mean(z, 0)
                ic_lm_names = [f"ICrit{x}" for x in range(NUM_ICRITICS)]
                # cc_lm_names = [f"CodeCrit{x}" for x in range(NUM_CODE_CRITICS)]

                noise = torch.normal(
                    torch.zeros([Z_DIM]), torch.ones([Z_DIM]) * target_std
                ).unsqueeze(0)

                wlog.update(
                    {
                        # "scaled depthwise gabian distance": scaled_depth_gabe_loss,
                        # "out of range": out_of_range_penalty,
                        # "scaled node stat loss": scaled_nsl,
                        # "mode loss": mode_loss.item(),
                        # "scaled mode loss": scaled_mode_loss.item(),
                        **{
                            f"z[{idx}] mean": node_mean[idx].item()
                            for idx in range(0, node_mean.shape[0], 64)
                        },
                        **{
                            f"z[{idx}] std": node_std[idx].item()
                            for idx in range(0, node_std.shape[0], 64)
                        },
                        # "icritic_multiloss": ic_multiloss.item(),
                        # **{f"image discernment {key} ": value.item() for key, value in zip(ic_lm_names, image_discernments)},
                        # **{f"scaled image discernment {key} ": value.item() for key, value in zip(ic_lm_names, image_discernments)},
                        # **{f"combo discernability progress {key} ": value.item() for key, value in zip(ic_lm_names, combo_disc_progresses)},
                        # # **{f"recon discernability progress {key} ": value.item() for key, value in zip(ic_lm_names, recon_disc_progresses)},
                        # **{f"icrit {key} Weight": value for key, value in zip(ic_lm_names, ic_weights)},
                        # **{f"code discernment {key} ": value.item() for key, value in zip(cc_lm_names, code_discernments)},
                        # **{f"scaled code discernment {key} ": value.item() for key, value in zip(cc_lm_names, code_discernments)},
                        # **{f"code discernability progress {key} ": value.item() for key, value in zip(cc_lm_names, code_disc_progresses)},
                        "dae_multilosses": dae_multiloss.item(),
                        # "dae_std_loss": std_loss.item(),
                        "z_entropy": entropy.item(),
                        "tmi": tmi.item(),
                        **{key: value for key, value in zip(dae_lm_names, dae_losses)},
                        **{
                            "DAE Scaled " + key: value
                            for key, value in zip(dae_lm_names, dae_scaled_losses)
                        },
                        **{
                            f"DAE {key} Weight": value
                            for key, value in zip(dae_lm_names, dae_weights)
                        },
                        # "cc_multilosses": cc_multiloss.item() * BATCH_MULTIPLIER,
                        # **{key: value for key, value in zip(cc_lm_names, cc_losses)},
                        # **{"cc Scaled " + key: value for key, value in zip(cc_lm_names, cc_scaled_losses)},
                        # **{f"cc {key} Weight": value for key, value in zip(cc_lm_names, cc_weights)},
                        "noise": wandb.Histogram(noise[0].tolist()),
                        "z": wandb.Histogram(uncut_z[0].tolist()),
                        **{
                            f"z[{idx}]": wandb.Histogram(uncut_z[:, idx].tolist())
                            for idx in range(0, Z_DIM, 256)
                        },
                    }
                )

                wandb.log(wlog)

                # if step % logInterval == 0:
                #     generated_image = generate_image()
                #     display_images(generated_image, cols=generated_image.shape[0], label=f"Generation Samples -- Epoch {epoch} Step {step}")
                ##################

                step += 1
                pbar.update()

            if len(DATA_CHUNKS) > 0:
                if chunk % 1 == 0:
                    ### save models ###
                    torch.save(dae, f"models/DAE_dae_tmi_{epoch}_animefaces.ckpt")
                    # for idx, c in enumerate(icritics):
                    #     torch.save(c, f"models/DAE_icritic{idx}_{epoch}_animefaces.ckpt")
                    # for idx, c in enumerate(code_critics):
                    #     torch.save(c, f"models/DAE_code_critic{idx}_{epoch}_animefaces.ckpt")

                if (chunk + 1) % 5 == 0:
                    print("\nRefreshing data.\n")
                    refresh_data()
                    print("\nRefreshed data.\n")
                else:
                    loader = torch.utils.data.DataLoader(  # this will make us use the same chunk again (a set of 3000 images)
                        dataset=dataset,
                        sampler=RandomSampler(dataset),
                        batch_size=TOTAL_BATCH_SIZE,
                        drop_last=True,
                        **kwargs,
                    )

                empty = False

                chunk += 1
                pbar.reset(total=len(loader))
                pbar.set_description(f"{epoch}, {chunk}")
            else:
                print("\nNo more chunks.\n")
                Still_Chunks = False
                refresh_chunks()
                refresh_data()
        ### save models ###
        torch.save(dae, f"models/DAE_dae_tmi_{epoch}_animefaces.ckpt")
        # for idx, c in enumerate(icritics):
        #     torch.save(c, f"models/DAE_icritic{idx}_{epoch}_animefaces.ckpt")
        # for idx, c in enumerate(code_critics):
        #     torch.save(c, f"models/DAE_code_critic{idx}_{epoch}_animefaces.ckpt")

        refresh_chunks()

### save final models ###
torch.save(dae, f"models/DAE_dae_tmi_FINAL_animefaces.ckpt")
# for idx, c in enumerate(icritics):
#     torch.save(c, f"models/DAE_icritic_FINAL_animefaces.ckpt")
# for idx, c in enumerate(code_critics):
#     torch.save(c, f"models/DAE_code_critic{idx}_FINAL_animefaces.ckpt")

# Generating a few samples
N = 16
sample = generate_image(N)
display_images(sample, n=N//4, cols=4)
