import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import model
import enhancement_loss
from dataloader import rgb2ycbcr,ycbcr2rgb
from math import exp


# class fusionloss(nn.Module):
#     def __init__(self):
#         super(fusionloss, self).__init__()
#         # 使用新的梯度计算方法
#         self.gradient = Gradient()  # 使用修改过的 Gradient 类
#         self.mse_loss = nn.MSELoss()
#         self.fuionnet = model.FusionNet().cuda()
#         self.angle = angle()
#         self.L_color = enhancement_loss.L_color()
#
#     def forward(self, vi_en_y, vi_en, ir, y_f, I_f):
#         # vi_en_y 和 ir 的梯度计算
#         vi_en_y_grad_x = self.gradient(vi_en_y, direction="x")
#         vi_en_y_grad_y = self.gradient(vi_en_y, direction="y")
#         ir_grad_x = self.gradient(ir, direction="x")
#         ir_grad_y = self.gradient(ir, direction="y")
#
#         # 计算梯度的最大值
#         vi_en_y_grad = torch.max(vi_en_y_grad_x, vi_en_y_grad_y)
#         ir_grad = torch.max(ir_grad_x, ir_grad_y)
#
#         # 计算 y_f 的梯度
#         y_f_grad_x = self.gradient(y_f, direction="x")
#         y_f_grad_y = self.gradient(y_f, direction="y")
#         y_f_grad = torch.max(y_f_grad_x, y_f_grad_y)
#
#         # 计算梯度损失
#         grad_loss = F.l1_loss(torch.max(vi_en_y_grad, ir_grad), y_f_grad)
#
#         # MSE 强度损失
#         max_init = torch.max(ir, vi_en_y)
#         image_loss = F.l1_loss(y_f, max_init)
#
#         # 颜色损失
#         color_loss = torch.mean(self.L_color(I_f))
#
#         # 总损失
#         # total_loss = 120 * image_loss + 10 * grad_loss + 0.05 * color_loss
#         total_loss = 120 * image_loss + 10 * grad_loss + 0.15 * color_loss
#
#         return total_loss, image_loss, grad_loss, color_loss

# class fusionloss(nn.Module):
#     def __init__(self):
#         super(fusionloss, self).__init__()
#         self.sobelconv=Sobelxy()
#         self.mse_loss = nn.MSELoss()
#         self.fuionnet = model.FusionNet().cuda()
#         self.angle=angle()
#         self.L_color=enhancement_loss.L_color()
#
#     # vi_en是三通道图片，ir是单通道灰度图，y_f是I_f的Y通道图，I_f由YCbCr组成
#     def forward(self,vi_en_y,vi_en,ir,y_f,I_f):#vi_en是RGB的
#         vi_en_y_gard=self.sobelconv(vi_en_y)
#         ir_gard=self.sobelconv(ir)
#         y_f_grad=self.sobelconv(y_f)
#         max_grad=torch.max(vi_en_y_gard,ir_gard)
#         grad_loss = F.l1_loss(max_grad,y_f_grad)
#
#         # MSE intensity Loss MSE强度损失函数
#         max_init = torch.max(ir,vi_en_y)
#         image_loss = F.l1_loss(y_f, max_init)
#         color_loss=torch.mean(self.L_color(I_f))
#         total_loss = 120*image_loss + 50*grad_loss + 0.05*color_loss
#         return total_loss,image_loss,grad_loss,color_loss

#用两个向量间的余弦值求的arccos





# 颜色角度损失：计算两个张量在RGB空间的角度差异
def angle_loss(a, b):
    # 计算 RGB 之间的角度（使用点积和反余弦公式）
    a_norm = F.normalize(a, p=2, dim=1)  # 对a进行归一化
    b_norm = F.normalize(b, p=2, dim=1)  # 对b进行归一化
    dot_product = torch.sum(a_norm * b_norm, dim=1, keepdim=True)
    angle = torch.acos(torch.clamp(dot_product, min=-1.0, max=1.0))  # 计算弧度
    return torch.mean(angle)  # 返回每个样本的平均角度


# 对比度损失：计算图像的对比度
def contrast_loss(x):
    # 计算图像的局部对比度
    mean_x = torch.mean(x, [2, 3], keepdim=True)  # 全局均值
    c = torch.sqrt(torch.mean((x - mean_x) ** 2, dim=[2, 3], keepdim=True))  # 计算局部对比度
    return c




class fusionloss(nn.Module):
    def __init__(self):
        super(fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.mse_loss = nn.MSELoss()
        self.fuionnet = model.FusionNet().cuda()
        # self.angle = angle_loss()
        self.L_color = enhancement_loss.L_color()
        # self.cmgc = GradientConsistencyLoss()

    def forward(self, vi_en_y, vi_en, ir, y_f, I_f,calc_loss=True):
        # 计算梯度损失
        vi_en_y_gard = self.sobelconv(vi_en_y)
        # vi_en = self.sobelconv(vi_en)
        ir_gard = self.sobelconv(ir)
        y_f_grad = self.sobelconv(y_f)
        max_grad = torch.max(vi_en_y_gard, ir_gard)
        grad_loss = F.l1_loss(max_grad, y_f_grad)

        #跨模态一致性损失
        # loss_gc = self.cmgc(vi_en, ir) if calc_loss else 0

        # MSE 强度损失
        max_init = torch.max(ir, vi_en_y)
        image_loss = F.l1_loss(y_f, max_init)

        # 颜色损失
        color_loss = torch.mean(self.L_color(I_f))

        # 像素级损失（保留红外强度）
        loss_pixel = F.l1_loss(I_f, ir) + F.l1_loss(I_f, vi_en_y)

        total_loss = 120 * image_loss + 10 * grad_loss + 0.05 * color_loss
        # + 0.2 * loss_gc

        return total_loss, image_loss, grad_loss , color_loss
            # ,loss_gc


class angle(nn.Module):
    def __init__(self):
        super(angle,self).__init__()
    def forward(self,a,b):
        vector = torch.mul(a,b) # 点乘
        up = torch.sum(vector)
        down = torch.sqrt(torch.sum(torch.pow(a,2))) * torch.sqrt(torch.sum(torch.pow(b,2)))
        theta = torch.acos(up/down)  # 弧度制
        return theta
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class SobelConv(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        sobel_kernel_x = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # 将 Sobel 核心扩展为 3 通道
        self.weightx = nn.Parameter(sobel_kernel_x.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        # 使用 groups=3 对每个通道分别执行 Sobel 卷积
        sobelx = F.conv2d(x, self.weightx, padding=1, groups=3)
        return sobelx
def get_per(img):
	fro_2_norm = torch.sum(torch.pow(img,2),dim=[1,2,3])
	loss=fro_2_norm / (225.0*225.0)
	return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  
    return window

def mssim(img1, img2, window_size=11):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2
    (_, channel, height, width) = img1.size()

    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2) 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def std(img,  window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma1

def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    return ssim.mean()


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


class GradientConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Sobel算子定义
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Create the Sobel kernels
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Set the kernel weights to the predefined Sobel kernels
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def gradient_map(self, x):
        # Ensure x and kernels are on the same device
        device = x.device  # Get device from the input tensor
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        if x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)  # Convert multi-channel to grayscale

        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

    def forward(self, vis_feat, ir_feat):
        """
        输入:
            vis_feat: 可见光特征图 [B,C,H,W]
            ir_feat: 红外特征图 [B,C,H,W]
        输出:
            loss: 梯度一致性损失值
        """
        grad_vis = self.gradient_map(vis_feat)
        grad_ir = self.gradient_map(ir_feat)
        return F.l1_loss(grad_vis, grad_ir)

