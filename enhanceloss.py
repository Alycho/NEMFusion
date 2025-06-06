import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

class LRetinex(nn.Module):
    def __init__(self):
        super(LRetinex, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, im, R, L):
        return self.mse(L * R, im) + self.mse(R, im / (L.detach()))


class LIlluination(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(LIlluination, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.mse = torch.nn.MSELoss()

    def forward(self, im, L):
        return self.cal_l_tv(L) + self.cal_l_init(im, L)

    def cal_l_init(self, im, L):
        im_max, _ = torch.max(im, dim=1, keepdim=True)
        return self.mse(im_max, L)

    def cal_l_tv(self, L):
        batch_size = L.size()[0]
        h_x = L.size()[2]
        w_x = L.size()[3]
        count_h = (L.size()[2] - 1) * L.size()[3]
        count_w = L.size()[2] * (L.size()[3] - 1)
        h_tv = torch.pow((L[:, :, 1:, :] - L[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((L[:, :, :, 1:] - L[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class LExp(nn.Module):
    def __init__(self):
        super(LExp, self).__init__()

    def forward(self, Iscb):
        mean = torch.nn.AvgPool2d(64)
        return torch.mean(torch.pow(mean(Iscb) - torch.FloatTensor([0.6]).to(Iscb.device), 2))


class LSC(nn.Module):
    def __init__(self):
        super(LSC, self).__init__()

    def forward(self, Iscb, R):
        R_gray = transforms.Grayscale()(R)
        return 1 - self.ssim_torch(Iscb, R_gray)

    def ssim_torch(self, im1, im2, L=1):
        K2 = 0.03
        C2 = (K2 * L) ** 2
        C3 = C2 / 2
        ux = torch.mean(im1)
        uy = torch.mean(im2)
        ox_sq = torch.var(im1)
        oy_sq = torch.var(im2)
        ox = torch.sqrt(ox_sq)
        oy = torch.sqrt(oy_sq)
        oxy = torch.mean((im1 - ux) * (im2 - uy))
        oxoy = ox * oy
        C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
        S = (oxy + C3) / (oxoy + C3)
        return S * C


class LHue(nn.Module):
    def __init__(self):
        super(LHue, self).__init__()
        self.mse = torch.nn.SmoothL1Loss()

    def forward(self, im_h, R_h):
        return self.mse(im_h, R_h)




class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()

        # Sobel kernels for x and y direction
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).unsqueeze(0).unsqueeze(
            0).cuda()  # shape [1, 1, 2, 2]
        self.smooth_kernel_y = self.smooth_kernel_x.transpose(2, 3)  # Transpose to get y direction kernel

    def forward(self, input_tensor, direction):
        # Choose kernel based on direction
        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        else:
            raise ValueError("Direction must be 'x' or 'y'")

        # Apply convolution to get the gradient
        gradient_orig = F.conv2d(input_tensor, kernel, stride=1, padding=1)

        # Compute the absolute value of the gradient
        gradient_abs = torch.abs(gradient_orig)

        # Normalize the gradient
        grad_min = torch.min(gradient_abs)
        grad_max = torch.max(gradient_abs)
        grad_norm = (gradient_abs - grad_min) / (grad_max - grad_min + 0.0001)

        return grad_norm



class GardLoss(nn.Module):
    def __init__(self):
        super(GardLoss, self).__init__()
        # 使用新的梯度计算方法
        self.gradient = Gradient()  # 使用修改过的 Gradient 类

        # 定义灰度转换
        self.to_gray = transforms.Grayscale(num_output_channels=1)  # 转换为单通道灰度图

    def forward(self, im1, im2):
        # 将图像转换为灰度图
        im1_gray = self.to_gray(im1)
        im2_gray = self.to_gray(im2)

        # vi_en_y 和 ir 的梯度计算
        im1_y_grad_x = self.gradient(im1_gray, direction="x")
        im1_y_grad_y = self.gradient(im1_gray, direction="y")
        im2_y_x = self.gradient(im2_gray, direction="x")
        im2_y_y = self.gradient(im2_gray, direction="y")

        # 计算梯度的最大值
        im1_y_grad = torch.max(im1_y_grad_x, im1_y_grad_y)
        im2_y_grad = torch.max(im2_y_x, im2_y_y)

        # 计算梯度损失
        grad_loss = F.l1_loss(im1_y_grad, im2_y_grad)

        return grad_loss


def gradient(input, axis):
    """
    计算图像的梯度。在 x 轴或者 y 轴上进行梯度计算。
    """
    if axis == "x":
        grad = F.conv2d(input, torch.tensor([[[[1, -1]]]], dtype=torch.float32).to(input.device), padding=(0, 1))
    elif axis == "y":
        grad = F.conv2d(input, torch.tensor([[[[1], [-1]]]], dtype=torch.float32).to(input.device), padding=(1, 0))
    return grad


def mutual_i_loss(input_I_low):
    """
    计算相互一致性损失
    :param input_I_low: 输入图像张量，假设形状为 (B, C, H, W)
    :return: 相互一致性损失
    """
    low_gradient_x = gradient(input_I_low, "x")
    x_loss = low_gradient_x * torch.exp(-10 * low_gradient_x)

    low_gradient_y = gradient(input_I_low, "y")
    y_loss = low_gradient_y * torch.exp(-10 * low_gradient_y)

    mutual_loss = torch.mean(x_loss + y_loss)
    return mutual_loss
# class gardloss(nn.Module):
#     def __init__(self):
#         super(gardloss, self).__init__()
#         # 使用新的梯度计算方法
#         self.gradient = Gradient()  # 使用修改过的 Gradient 类
#         # self.mse_loss = nn.MSELoss()
#
#     def forward(self, im1_y, im2_y):
#         # vi_en_y 和 ir 的梯度计算
#         im1_y_grad_x = self.gradient(im1_y, direction="x")
#         im1_y_grad_y = self.gradient(im1_y, direction="y")
#         im2_y_x = self.gradient(im2_y, direction="x")
#         im2_y_y = self.gradient(im2_y, direction="y")
#
#         # 计算梯度的最大值
#         im1_y_grad = torch.max(im1_y_grad_x, im1_y_grad_y)
#         im2_y_grad = torch.max(im2_y_x, im2_y_y)
#
#         # # 计算 y_f 的梯度
#         # y_f_grad_x = self.gradient(y_f, direction="x")
#         # y_f_grad_y = self.gradient(y_f, direction="y")
#         # y_f_grad = torch.max(y_f_grad_x, y_f_grad_y)
#
#         # 计算梯度损失
#         grad_loss = F.l1_loss(torch.max(im1_y_grad, im2_y_grad))
#
#         # # MSE 强度损失
#         # max_init = torch.max(im1_y_grad, im2_y_grad)
#         # image_loss = F.l1_loss(y_f, max_init)
#
#         return   grad_loss

