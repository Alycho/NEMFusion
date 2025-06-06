import os
import cv2
import numpy as np
import glob
import random

import torch
import torch.utils.data as data

from PIL import Image
from matplotlib import transforms
from torchvision import transforms
to_tensor = transforms.Compose([transforms.ToTensor()])

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)  # 列出目录中的所有文件
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))  # 查找所有 .bmp 文件
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))  # 查找 .tif 文件并添加到列表中
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))  # 查找 .jpg 文件
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))  # 查找 .png 文件
    data.sort()  # 排序文件路径列表
    filenames.sort()  # 排序文件名列表
    return data, filenames  # 返回文件路径和文件名

class CustomDataLoader:
    def __init__(self, visible_dir, infrared_dir, image_size=(256, 256), augment=False):
        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.image_size = image_size
        self.augment = augment
        self.visible_images = []
        for dir in visible_dir:
            self.visible_images.extend(sorted(glob.glob(os.path.join(dir, '*.jpg'))))
        self.infrared_images = []
        for dir in infrared_dir:
            self.infrared_images.extend(sorted(glob.glob(os.path.join(dir, '*.jpg'))))

        # 确保可见光和红外图像数量相同
        if len(self.visible_images) != len(self.infrared_images):
            raise ValueError("The number of visible images and infrared images must be the same.")

    def __len__(self):
        return len(self.visible_images)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to read: {image_path}")
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0  # 归一化
        return image

    def apply_clahe(self, image):
        # 将图像转换为灰度图，CLAHE 通常应用于灰度图
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)
        clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)  # 转换回彩色图像格式
        clahe_image = clahe_image.astype(np.float32) / 255.0  # 再次归一化
        return clahe_image

    def augment_image(self, image):
        if random.random() > 0.5:  # 随机翻转
            image = cv2.flip(image, 1)
        return image

    def __getitem__(self, index):
        visible_image = self.load_image(self.visible_images[index])
        infrared_image = self.load_image(self.infrared_images[index])

        # Apply augmentation if needed
        if self.augment:
            visible_image = self.augment_image(visible_image)
            infrared_image = self.augment_image(infrared_image)

        # Apply CLAHE to the visible image (or both, depending on your needs)
        clahe_image = self.apply_clahe(visible_image)

        # 转换图像形状为 (C, H, W)
        visible_image = np.transpose(visible_image, (2, 0, 1))  # 从 (H, W, C) 转为 (C, H, W)
        infrared_image = np.transpose(infrared_image, (2, 0, 1))  # 从 (H, W, C) 转为 (C, H, W)
        clahe_image = np.transpose(clahe_image, (2, 0, 1))  # 从 (H, W, C) 转为 (C, H, W)

        return visible_image, infrared_image, clahe_image  # 返回三个值


import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

class LoadDataset(data.Dataset):
    # 增强数据加载器
    def __init__(self, data_dir, tr, img_size=None):
        self.data_dir = data_dir
        self.transform = tr
        self.img_size = img_size
        self.img_list = self.load_imglist()  # 只调用一次

    def __getitem__(self, idx):
        img, img_clahe, image_name = self.load_img(self.img_list[idx])
        return img, img_clahe


    def __len__(self):
        return len(self.img_list)

    def load_img(self, img_path):
        img = Image.open(img_path)
        if self.img_size is not None:
            img_norm = img.resize((self.img_size[0], self.img_size[1]), Image.LANCZOS)

        else:
            img_norm = img

        img_norm = self.transform(img_norm)
        img_clahe_norm = self.img_HE(img_norm)  # 获取 CLAHE 图像
        img_name = os.path.basename(img_path)
        return img_norm, img_clahe_norm, img_name

    def load_imglist(self):
        img_list = glob.glob(os.path.join(self.data_dir, '*/*.*'))
        if len(img_list) == 0:
            img_list = glob.glob(os.path.join(self.data_dir, '*.*'))
        return img_list

    def img_HE(self, tensor):
        def clahe(im):

            ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
            channels = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
            channels[0] = clahe.apply(channels[0])  # 对 Y 通道应用 CLAHE
            ycrcb = cv2.merge(channels)  # 合并通道
            im = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)  # 转换回 RGB
            return im

        # 将 tensor 转为 numpy 数组并应用 CLAHE
        im_pros = tensor.mul(255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        # 应用 CLAHE 和滤波
        im_pros = clahe(im_pros.copy())
        im_pros = cv2.bilateralFilter(im_pros, 5, 10, 10)
        im_pros = cv2.cvtColor(im_pros, cv2.COLOR_RGB2GRAY)
        im_pros = im_pros.astype(np.uint8)

        # 转回为 tensor
        im_clahe = transforms.ToTensor()(im_pros)
        return im_clahe


class LoadDataset1(data.Dataset):
    # 增强数据加载器
    def __init__(self, data_dir, tr, img_size=None):
        self.data_dir = data_dir
        self.transform = tr
        self.img_size = img_size
        self.img_list = self.load_imglist()  # 只调用一次

    def __getitem__(self, idx):
        img, img_clahe, image_name = self.load_img(self.img_list[idx])
        return img, img_clahe, image_name  # 返回名称

    def __len__(self):
        return len(self.img_list)

    def load_img(self, img_path):
        img = Image.open(img_path)
        if self.img_size is not None:
            img_norm = img.resize((self.img_size[0], self.img_size[1]), Image.LANCZOS)
        else:
            img_norm = img

        img_norm = self.transform(img_norm)
        img_clahe_norm = self.img_HE(img_norm)  # 获取 CLAHE 图像
        img_name = os.path.basename(img_path)  # 获取图像名称
        return img_norm, img_clahe_norm, img_name  # 返回名称

    def load_imglist(self):
        img_list = glob.glob(os.path.join(self.data_dir, '*/*.*'))
        if len(img_list) == 0:
            img_list = glob.glob(os.path.join(self.data_dir, '*.*'))
        return img_list

    def img_HE(self, tensor):
        def clahe(im):
            ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
            channels = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
            channels[0] = clahe.apply(channels[0])  # 对 Y 通道应用 CLAHE
            ycrcb = cv2.merge(channels)  # 合并通道
            im = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)  # 转换回 RGB
            return im

        # 将 tensor 转为 numpy 数组并应用 CLAHE
        im_pros = tensor.mul(255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        # 应用 CLAHE 和滤波
        im_pros = clahe(im_pros.copy())
        im_pros = cv2.bilateralFilter(im_pros, 5, 10, 10)
        im_pros = cv2.cvtColor(im_pros, cv2.COLOR_RGB2GRAY)
        im_pros = im_pros.astype(np.uint8)

        # 转回为 tensor
        im_clahe = transforms.ToTensor()(im_pros)
        return im_clahe







class fusion_dataset_loader(data.Dataset):
    def __init__(self, ir_path='ours_dataset_240/train_set/infrared', vi_path='ours_dataset_240/train_set/vi_en',img_size=(256,256)):
        super(fusion_dataset_loader, self).__init__()
        self.img_size = img_size # 图像尺寸

        # 数据路径和文件名列表
        self.filepath_vis, self.filenames_vis = prepare_data_path(vi_path)  # 可见光图像路径
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)  # 红外图像路径
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))  # 确保图像对数相同

    def __getitem__(self, index):
            vis_path = self.filepath_vis[index]  # 获取可见光图像路径
            ir_path = self.filepath_ir[index]  # 获取红外图像路径

        # 处理可见光图像
            image_vis = Image.open(vis_path)  # 打开可见光图像
            image_vis = image_vis.resize((self.size, self.size), Image.ANTIALIAS)  # 调整图像尺寸
            image_vis = np.array(image_vis)  # 转换为 NumPy 数组
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                (2, 0, 1)) / 255.0)  # 转换为 Tensor 格式并归一化

            # 处理红外图像
            image_inf = Image.open(ir_path)  # 打开红外图像
            image_inf = image_inf.resize((self.size, self.size), Image.ANTIALIAS)  # 调整尺寸
            image_inf = np.array(image_inf)  # 转换为 NumPy 数组

            # 确保图像为单通道，并添加通道维度
            if len(image_inf.shape) == 2:  # 灰度图
                image_ir = np.expand_dims(image_inf, axis=0)  # 添加通道维度
            else:  # RGB 图像，选择其中一个通道
                image_ir = image_inf[:, :, 0]  # 选择第一个通道
                image_ir = np.expand_dims(image_ir, axis=0)  # 添加通道维度

            image_ir = np.asarray(image_ir, dtype=np.float32) / 255.0  # 归一化
            image_vis = torch.tensor(image_vis)  # 转换为 PyTorch Tensor
            image_ir = torch.tensor(image_ir)  # 转换为 PyTorch Tensor

            return (image_vis, image_ir)  # 返回可见光和红外图像对

    def __len__(self):
        return self.length  # 返回数据集长度




class fusion_dataset_loader_eval(data.Dataset):
    def __init__(self, ir_path, vi_path, transform=to_tensor):
        super().__init__()
        # 使用手动指定的路径
        self.inf_path = ir_path  # 红外图像路径
        self.vis_path = vi_path  # 可见光图像路径
        self.name_list = os.listdir(self.inf_path)  # 获取红外图像文件名列表
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获取图像名
        inf_image = Image.open(os.path.join(self.inf_path, name))  # 打开红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name))  # 打开可见光图像
        ir_image = self.transform(inf_image)  # 转换为 Tensor
        vis_image = self.transform(vis_image)  # 转换为 Tensor
        return vis_image, ir_image, name  # 返回可见光和红外图像及其文件名

    def __len__(self):
        return len(self.name_list)  # 返回数据集长度





# 测试集类：用于加载融合数据集（测试用）
class fusion_dataset_loader_test(data.Dataset):
    def __init__(self, ir_path, vi_path, transform=to_tensor):
        super().__init__()
        # 使用手动指定的路径
        self.inf_path = ir_path  # 红外图像路径
        self.vis_path = vi_path  # 可见光图像路径
        self.name_list = os.listdir(self.inf_path)  # 获取红外图像文件名列表
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获取图像名
        inf_image = Image.open(os.path.join(self.inf_path, name))  # 打开红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name))  # 打开可见光图像
        ir_image = self.transform(inf_image)  # 转换为 Tensor
        vis_image = self.transform(vis_image)  # 转换为 Tensor
        return vis_image, ir_image, name  # 返回可见光和红外图像及其文件名

    def __len__(self):
        return len(self.name_list)  # 返回数据集长度





def rgb2ycbcr(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # 调整图像维度并展平
    R = im_flat[:, 0]  # 提取 R 通道
    G = im_flat[:, 1]  # 提取 G 通道
    B = im_flat[:, 2]  # 提取 B 通道
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # 计算 Y 分量
    Cr = (R - Y) * 0.713 + 0.5  # 计算 Cr 分量
    Cb = (B - Y) * 0.564 + 0.5  # 计算 Cb 分量

    # 限制 Y, Cr, Cb 的取值范围
    Y = torch.clamp(Y, min=0., max=1.0)
    Cr = torch.clamp(Cr, min=0., max=1.0)
    Cb = torch.clamp(Cb, min=0., max=1.0)
    Y = torch.unsqueeze(Y, 1)  # 添加通道维度
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    # 将 Y, Cr, Cb 合并为一个 Tensor
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = temp.reshape(
            list(input_im.size())[0],  # batch size
            list(input_im.size())[2],  # 宽度
            list(input_im.size())[3],  # 高度
            3).transpose(1, 3).transpose(2, 3)
    return out  # 返回转换后的图像

# 函数：将 YCbCr 图像转换为 RGB 颜色空间
def ycbcr2rgb(input_im):
    B, C, W, H = input_im.shape  # 获取输入图像的维度
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # 展平图像
    # 定义 YCbCr 到 RGB 转换矩阵和偏移
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()  # 应用矩阵变换
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3).cuda()  # 恢复原来的维度顺序
    out = torch.clamp(out, min=0., max=1.0)  # 限制 RGB 值范围
    return out  # 返回转换后的图像


# 函数：应用 CLAHE (对比度受限的自适应直方图均衡化) 处理图像
def clahe(image, batch_size):
    image = image.cpu().detach().numpy()  # 将 Tensor 移到 CPU 并转换为 NumPy 数组
    results = []  # 保存处理后的图像
    for i in range(batch_size):
        img = np.squeeze(image[i:i + 1, :, :, :])  # 移除多余的维度
        out = np.array(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), dtype='uint8')  # 归一化图像并转换为 8 位整数
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))  # 创建 CLAHE 对象
        result = clahe.apply(out)[np.newaxis][np.newaxis]  # 应用 CLAHE 并扩展维度
        results.append(result)  # 保存结果
    results = np.concatenate(results, axis=0)  # 合并结果
    image_hist = (results / 255.0).astype(np.float32)  # 归一化处理后的图像
    image_hist = torch.from_numpy(image_hist).cuda()  # 转换为 PyTorch Tensor 并移动到 GPU
    return image_hist  # 返回处理后的图像
