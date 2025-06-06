import cv2
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import tensorflow as tf
from dual_attention_fusion_module import attention_fusion_weight




# ULSAM: Ultra-Lightweight Subspace Attention Module for Compact Convolutional Neural Networks(WACV20)

class SubSpace(nn.Module):
    def __init__(self, nin: int) -> None:
        super(SubSpace, self).__init__()

        # 动态调整 groups 使其与输入通道数一致
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=1)  # Softmax applied over the channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Input shape: {x.shape}")

        nin = x.size(1)  # 动态获取输入的通道数

        # 如果通道数与 groups 不一致，则调整 groups
        self.conv_dws.groups = nin

        out = self.conv_dws(x)  # Depthwise convolution
        # print(f"After conv_dws: {out.shape}")

        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        # print(f"After conv_point: {out.shape}")

        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out * x  # Element-wise multiplication
        out = out + x  # Skip connection

        return out


class ULSAM(nn.Module):
    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int) -> None:
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for _ in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the input tensor into subspaces along the channel dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out


class juanji_sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(juanji_sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class get_layer1(nn.Module):
    def __init__(self, num_channels, growth):
        super(get_layer1, self).__init__()
        self.conv_1 = ConvLeakyRelu2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1,
                                      bias=True)
        self.conv_2 = ConvLeakyRelu2d(in_channels=32, out_channels=growth, kernel_size=3, stride=1, padding=1,
                                      bias=True)

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)
        return x1


class denselayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(denselayer, self).__init__()
        self.conv_1 = ConvLeakyRelu2d(in_channels=num_channels, out_channels=growth, kernel_size=3, stride=1, padding=1,
                                      bias=True)
        self.sobel = juanji_sobelxy(num_channels)
        self.sobel_conv = nn.Conv2d(num_channels, growth, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.sobel(x)
        x2 = self.sobel_conv(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class ConvLeakyRelu2d(nn.Module):
    # convolution + leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.2)



class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DepScp_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepScp_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        # print(f'Input shape: {x.shape}')  # 添加调试输出
        x1 = self.depth_conv(x)
        out = self.point_conv(x1)
        return out


class SCR(nn.Module):
    """
    用于恢复退化结构和纹理的可学习分支
    低光图像本身的特点
    """

    def __init__(self):
        super(SCR, self).__init__()
        filters = 16
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.ds_conv1 = DepScp_Conv(3, filters)
        self.ds_conv2 = DepScp_Conv(filters, filters)
        self.ds_conv3 = DepScp_Conv(filters, filters)
        self.ds_conv4 = DepScp_Conv(filters, 1)

    def forward(self, x, x_h):
        """
       @param x: 低光输入
       @param x_h: 直方图均衡化后的 x
       @return: 恢复后的 x_h

        """
        x1 = self.relu(self.ds_conv1(x))
        x2 = self.relu(self.ds_conv2(x1))
        x3 = self.relu(self.ds_conv3(x2))
        alpha = self.sigmoid(self.ds_conv4(x3)) + 1
        gam_batch = self.cal_gam_val(x.detach())  # adaptive gamma correction
        # 假设 x_h 已经在正确的设备上
        device = x_h.device

        # 确保 gam_batch 和 alpha 也在同一个设备上
        gam_batch = gam_batch.to(device)
        alpha = alpha.to(device)

        x_scr = torch.pow(x, gam_batch) + (alpha * x)
        x_h_scr = torch.pow(x_h, gam_batch) + (alpha * x_h)
        # print(f"x shape: {x.shape}, x_h_scr shape: {x_h_scr.shape}")
        return x_scr, x_h_scr

    def cal_gam_val(self, im):
        gam_batch = torch.tensor([])
        for i in range(im.shape[0]):
            im_np = im[i].mul(255).byte()
            im_np = im_np.detach().cpu().numpy().transpose(1, 2, 0)
            mean_gray = np.mean(np.max(im_np, axis=2))

            if mean_gray < 255 / 2:
                if mean_gray == 0:  # set default value, avoid divide by zero
                    gam_batch = torch.cat([gam_batch, torch.tensor([0.25])], dim=0)
                else:  # cal adaptive gamma value
                    gam = math.log10(0.5) / math.log10(mean_gray / 255.0)
                    gam_batch = torch.cat([gam_batch, torch.tensor([min(1, gam)])], dim=0)
            else:  # skip if input image is bright enough
                gam_batch = torch.cat([gam_batch, torch.tensor([1])], dim=0)
        gam_batch = gam_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(im.device)
        return gam_batch

class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        filters = 64
        self.SCR = SCR()
        self.ULSAM1 = ULSAM(nin=3, nout=3, h=64, w=64, num_splits=1)
        self.ULSAM2 = ULSAM(nin=1, nout=1, h=64, w=64, num_splits=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = Conv(4, filters)
        self.conv2 = Conv(filters, filters)
        self.conv3 = Conv(filters, filters)
        self.conv4 = Conv(filters, filters)
        self.conv5 = Conv(filters * 2, filters)
        self.conv6 = Conv(filters * 2, filters)
        self.conv7 = Conv(filters * 2, 4)

    def forward(self, x, x_h):
        x_scr, x_h_scr = self.SCR(x, x_h)
        # print(f"x_scr shape: {x_scr.shape}, x_h_scr shape: {x_h_scr.shape}")
        #
        # print(f"x shape: {x.shape}, x_h_scr shape: {x_h_scr.shape}")
        x = self.ULSAM1(x)  # 增强后的 x
        x_h_scr = self.ULSAM2(x_h_scr)  # 增强后的 x_h_scr
        x_out = torch.cat([x, x_h_scr], dim=1)
        # print(f"x_out shape: {x_out.shape}")
        # print(f"Before conv1, input shape: {x_out.shape}")
        x1 = self.relu(self.conv1(x_out))
        # print(f"x1 shape: {x1.shape}")

        x2 = self.relu(self.conv2(x1))
        # print(f"x2 shape: {x2.shape}")

        x3 = self.relu(self.conv3(x2))
        # print(f"x3 shape: {x3.shape}")

        x4 = self.relu(self.conv4(x3))
        # print(f"x4 shape: {x4.shape}")

        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        # print(f"x5 shape: {x5.shape}")

        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        # print(f"x6 shape: {x6.shape}")

        x7 = self.sigmoid(self.conv7(torch.cat([x1, x6], 1)))
        # print(f"x7 shape: {x7.shape}")

        R, L = torch.split(x7, 3, 1)
        # print(f"R shape: {R.shape}, L shape: {L.shape}")

        # extract hue information of images from HSV color space
        R_hue = self.rgb2hsv(R)[:, 0, :, :] / 360.0
        # print(f"R_hue shape: {R_hue.shape}")

        # 检查 R_hue 的形状
        if R_hue.dim() == 3:
            R_hue = R_hue.unsqueeze(-1)  # 添加宽度维度
        # print(f"R_hue shape after unsqueeze: {R_hue.shape}")

        # 确保填充大小不会超过输入的高度和宽度
        batch_size, channels, height, width = R_hue.shape
        kernel_size = 7
        padding = (1, 1)

        # 确保填充大小不超过输入的维度
        if padding[1] >= width:
            padding = (padding[0], width - 1)  # 调整为适当的填充大小

        # 对 R_hue 进行填充
        # print(f"Padding before pad: {padding}, R_hue shape: {R_hue.shape}")
        R_hue = F.pad(R_hue, (padding[1], padding[1], padding[0], padding[0]), mode='reflect')
        # print(f"R_hue shape after padding: {R_hue.shape}")

        # 进行高斯模糊
        kernel_size = min(R_hue.shape[2], R_hue.shape[3])  # 确保内核大小不超过维度
        R_hue = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1, 1))(R_hue)


        x_hue = self.rgb2hsv(x)[:, 0, :, :] / 360.0
        x_hue = F.pad(x_hue.unsqueeze(1), (padding[1], padding[1], padding[0], padding[0]), mode='reflect')
        x_hue = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1, 1))(x_hue)
        # print(f"x_hue shape after GaussianBlur: {x_hue.shape}")

        return x_scr, x_h_scr, R, L, R_hue, x_hue

    def rgb2hsv(self, input, epsilon=1e-10):
        assert (input.shape[1] == 3)  # 确保输入的通道数为 3
        r, g, b = input[:, 0], input[:, 1], input[:, 2]
        max_rgb, argmax_rgb = input.max(1)
        min_rgb, argmin_rgb = input.min(1)
        max_min = max_rgb - min_rgb + epsilon

        h1 = 60.0 * (g - r) / max_min + 60.0
        h2 = 60.0 * (b - g) / max_min + 180.0
        h3 = 60.0 * (r - b) / max_min + 300.0

        h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
        s = max_min / (max_rgb + epsilon)
        v = max_rgb
        return torch.stack((h, s, v), dim=1)




# class DecomNet(nn.Module):
#     def __init__(self):
#         super(DecomNet, self).__init__()
#         filters = 64
#         self.SCR = SCR()
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#         self.conv1 = Conv(4, filters)
#         self.conv2 = Conv(filters, filters)
#         self.conv3 = Conv(filters, filters)
#         self.conv4 = Conv(filters, filters)
#         self.conv5 = Conv(filters * 2, filters)
#         self.conv6 = Conv(filters * 2, filters)
#         self.conv7 = Conv(filters * 2, 4)
#
#         # Add ULSAM after convolution layers
#         self.ulsam1 = ULSAM(nin=filters, nout=filters, h=256, w=256, num_splits=4)
#         self.ulsam2 = ULSAM(nin=filters, nout=filters, h=256, w=256, num_splits=4)
#         self.ulsam3 = ULSAM(nin=filters, nout=filters, h=256, w=256, num_splits=4)
#         self.ulsam4 = ULSAM(nin=filters, nout=filters, h=256, w=256, num_splits=4)
#         self.ulsam5 = ULSAM(nin=filters * 2, nout=filters * 2, h=256, w=256, num_splits=4)
#         self.ulsam6 = ULSAM(nin=filters * 2, nout=filters * 2, h=256, w=256, num_splits=4)
#
#     def forward(self, x, x_h):
#         x_scr, x_h_scr = self.SCR(x, x_h)
#
#         # Concatenate the inputs
#         x_out = torch.cat([x, x_h_scr], dim=1)
#
#         # Pass through conv layers and apply ULSAM after each
#         x1 = self.relu(self.conv1(x_out))
#         x1 = self.ulsam1(x1)  # Apply ULSAM after the first conv layer
#
#         x2 = self.relu(self.conv2(x1))
#         x2 = self.ulsam2(x2)  # Apply ULSAM after the second conv layer
#
#         x3 = self.relu(self.conv3(x2))
#         x3 = self.ulsam3(x3)  # Apply ULSAM after the third conv layer
#
#         x4 = self.relu(self.conv4(x3))
#         x4 = self.ulsam4(x4)  # Apply ULSAM after the fourth conv layer
#
#         x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
#         x5 = self.ulsam5(x5)  # Apply ULSAM after the fifth conv layer
#
#         x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
#         x6 = self.ulsam6(x6)  # Apply ULSAM after the sixth conv layer
#
#         x7 = self.sigmoid(self.conv7(torch.cat([x1, x6], 1)))
#
#         R, L = torch.split(x7, 3, 1)
#
#         # Additional processing (e.g., hue extraction)...
#         R_hue = self.rgb2hsv(R)[:, 0, :, :] / 360.0
#         R_hue = F.pad(R_hue.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
#         R_hue = transforms.GaussianBlur(kernel_size=7, sigma=(1, 1))(R_hue)
#
#         x_hue = self.rgb2hsv(x)[:, 0, :, :] / 360.0
#         x_hue = F.pad(x_hue.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
#         x_hue = transforms.GaussianBlur(kernel_size=7, sigma=(1, 1))(x_hue)
#
#         return x_scr, x_h_scr, R, L, R_hue, x_hue
#
#     def rgb2hsv(self, input, epsilon=1e-10):
#         assert (input.shape[1] == 3)  # 确保输入的通道数为 3
#         r, g, b = input[:, 0], input[:, 1], input[:, 2]
#         max_rgb, argmax_rgb = input.max(1)
#         min_rgb, argmin_rgb = input.min(1)
#         max_min = max_rgb - min_rgb + epsilon
#
#         h1 = 60.0 * (g - r) / max_min + 60.0
#         h2 = 60.0 * (b - g) / max_min + 180.0
#         h3 = 60.0 * (r - b) / max_min + 300.0
#
#         h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
#         s = max_min / (max_rgb + epsilon)
#         v = max_rgb
#         return torch.stack((h, s, v), dim=1)


# class DecomNet(nn.Module):
#     def __init__(self):
#         super(DecomNet, self).__init__()
#         filters = 64
#         self.SCR = SCR()
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#         self.conv1 = Conv(4, filters)
#         self.conv2 = Conv(filters, filters)
#         self.conv3 = Conv(filters, filters)
#         self.conv4 = Conv(filters, filters)
#         self.conv5 = Conv(filters * 2, filters)
#         self.conv6 = Conv(filters * 2, filters)
#         self.conv7 = Conv(filters * 2, 4)
#
#     def forward(self, x, x_h):
#         x_scr, x_h_scr = self.SCR(x, x_h)
#         # print(f"x_scr shape: {x_scr.shape}, x_h_scr shape: {x_h_scr.shape}")
#         #
#         # print(f"x shape: {x.shape}, x_h_scr shape: {x_h_scr.shape}")
#
#         x_out = torch.cat([x, x_h_scr], dim=1)
#         # print(f"x_out shape: {x_out.shape}")
#         # print(f"Before conv1, input shape: {x_out.shape}")
#         x1 = self.relu(self.conv1(x_out))
#         # print(f"x1 shape: {x1.shape}")
#
#         x2 = self.relu(self.conv2(x1))
#         # print(f"x2 shape: {x2.shape}")
#
#         x3 = self.relu(self.conv3(x2))
#         # print(f"x3 shape: {x3.shape}")
#
#         x4 = self.relu(self.conv4(x3))
#         # print(f"x4 shape: {x4.shape}")
#
#         x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
#         # print(f"x5 shape: {x5.shape}")
#
#         x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
#         # print(f"x6 shape: {x6.shape}")
#
#         x7 = self.sigmoid(self.conv7(torch.cat([x1, x6], 1)))
#         # print(f"x7 shape: {x7.shape}")
#
#         R, L = torch.split(x7, 3, 1)
#         # print(f"R shape: {R.shape}, L shape: {L.shape}")
#
#         # extract hue information of images from HSV color space
#         R_hue = self.rgb2hsv(R)[:, 0, :, :] / 360.0
#         # print(f"R_hue shape: {R_hue.shape}")
#
#         # 检查 R_hue 的形状
#         if R_hue.dim() == 3:
#             R_hue = R_hue.unsqueeze(-1)  # 添加宽度维度
#         # print(f"R_hue shape after unsqueeze: {R_hue.shape}")
#
#         # 确保填充大小不会超过输入的高度和宽度
#         batch_size, channels, height, width = R_hue.shape
#         kernel_size = 7
#         padding = (1, 1)
#
#         # 确保填充大小不超过输入的维度
#         if padding[1] >= width:
#             padding = (padding[0], width - 1)  # 调整为适当的填充大小
#
#         # 对 R_hue 进行填充
#         # print(f"Padding before pad: {padding}, R_hue shape: {R_hue.shape}")
#         R_hue = F.pad(R_hue, (padding[1], padding[1], padding[0], padding[0]), mode='reflect')
#         # print(f"R_hue shape after padding: {R_hue.shape}")
#
#         # 进行高斯模糊
#         kernel_size = min(R_hue.shape[2], R_hue.shape[3])  # 确保内核大小不超过维度
#         R_hue = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1, 1))(R_hue)
#
#
#         x_hue = self.rgb2hsv(x)[:, 0, :, :] / 360.0
#         x_hue = F.pad(x_hue.unsqueeze(1), (padding[1], padding[1], padding[0], padding[0]), mode='reflect')
#         x_hue = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1, 1))(x_hue)
#         # print(f"x_hue shape after GaussianBlur: {x_hue.shape}")
#
#         return x_scr, x_h_scr, R, L, R_hue, x_hue
#
#     def rgb2hsv(self, input, epsilon=1e-10):
#         assert (input.shape[1] == 3)  # 确保输入的通道数为 3
#         r, g, b = input[:, 0], input[:, 1], input[:, 2]
#         max_rgb, argmax_rgb = input.max(1)
#         min_rgb, argmin_rgb = input.min(1)
#         max_min = max_rgb - min_rgb + epsilon
#
#         h1 = 60.0 * (g - r) / max_min + 60.0
#         h2 = 60.0 * (b - g) / max_min + 180.0
#         h3 = 60.0 * (r - b) / max_min + 300.0
#
#         h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
#         s = max_min / (max_rgb + epsilon)
#         v = max_rgb
#         return torch.stack((h, s, v), dim=1)


class EnhStage():
    def __init__(self):
        super().__init__()

    def enh(self, im, R, L, im_h_scr, scale=0.85):
        mean = torch.mean(im)
        gam = math.log10(scale) / math.log10(mean)
        out = self.guided_filter(R, im_h_scr) * torch.pow(L, gam)
        return out

    def guided_filter(self, im, guide):
        im_np = im.mul(255).squeeze(0).byte()
        im_np = im_np.detach().cpu().numpy().transpose(1, 2, 0)
        guide_np = guide.squeeze(0).mul(255).byte()
        guide_np = guide_np.detach().cpu().numpy().transpose(1, 2, 0)
        im_dns = cv2.ximgproc.guidedFilter(guide_np, im_np, 1, 50, -1)
        im_dns = transforms.ToTensor()(im_dns).to(im.device)
        return im_dns






