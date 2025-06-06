import os

from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU运行
from torch.autograd import Variable
import argparse
import datetime
import time
import math
import logging
import os.path as osp
import torch
import dataloader
import model
import enhancement_loss
from fusion_loss import fusionloss, final_ssim
from logger import setup_logger
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader import rgb2ycbcr, ycbcr2rgb
from torchvision import transforms
from tqdm import tqdm
from thop import profile
import warnings
import fusion_model
import enhance_model
import torch.nn.init as init

warnings.filterwarnings('ignore')

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):  # 对卷积层进行初始化
        print(f"Initializing Conv2d: {m}")
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用 He 初始化
        if m.bias is not None:
            init.zeros_(m.bias)  # 偏置初始化为零
    elif isinstance(m, nn.BatchNorm2d):  # 对BatchNorm层进行初始化
        print(f"Initializing BatchNorm2d: {m}")
        init.ones_(m.weight)  # 权重初始化为1
        init.zeros_(m.bias)   # 偏置初始化为0
    elif isinstance(m, nn.Linear):  # 对全连接层进行初始化
        print(f"Initializing Linear: {m}")
        init.xavier_normal_(m.weight)  # 使用 Xavier 初始化
        if m.bias is not None:
            init.zeros_(m.bias)  # 偏置初始化为0
def weights_init(m):
    # 获得nn.module的名字，初始化权重
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # mean=0.0,std=0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


def train_fusion(i, logger=None):  # 再增强与融合网络的训练函数
    modelpth = './model/CDFF'
    modelpth = osp.join(modelpth, str(i + 1))
    os.makedirs(modelpth, mode=0o777, exist_ok=True)
    fusion_batch_size = 5
    # enhance_model_path = osp.join(os.getcwd(), 'model', str(i + 1), 'epoch_90.pth')
    enhance_model_path = 'checkpoint/epoch_90.pth'
    n_workers = 4

    train_transforms = transforms.Compose([transforms.RandomCrop((128, 128)),
                                                                                      transforms.ToTensor()])
    ds = dataloader.fusion_dataset_loader('train',train_transforms)  # load training data
    dl = torch.utils.data.DataLoader(ds, batch_size=fusion_batch_size, shuffle=True, num_workers=n_workers,
                                                                               pin_memory=False)
    # dl是增强之后的vi_en和ir
    net = model.FusionNet()
    if i == 0:
        net.apply(weights_init)
    if i > 0:  # 第二次训练加载上一次训练的字典
        load_path = './model/CDFF'
        load_path = osp.join(load_path, str(i), 'fusion_model_best.pth')
        net.load_state_dict(torch.load(load_path))
        print('Load Pre-trained Fusion Model:{}!'.format(load_path))  # 加载之前训练的fusion模型
    net.train()
    enhancemodel = enhance_model.DecomNet().cuda()  # 亮度调整网络
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    enhancemodel.eval()
    enh_module = enhance_model.EnhStage()
    lr_start = 5e-4
    optim = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=0.0001)
    criteria_fusion = fusionloss()
    st = glob_st = time.time()
    epoch = 50
    grad_step = 5.0  # gradient accumulation to increase batch_size
    dl.n_iter = len(dl)
    # 保存最小损失对应的模型
    best_loss = float('inf')  # 初始时设置最小损失为无穷大
    best_model = None  # 用来保存最小损失时的模型
    best_epoch = 0  # 记录最佳模型所在的 epoch

    for epo in range(0, epoch):
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** ((epo / 5) + 1)  # 迭代变更初始学习率
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_vis, image_ir, image_vis_h) in enumerate(dl):
            net.train().cuda()
            image_vis = Variable(image_vis, requires_grad=True).float().cuda()
            image_vis_h = Variable(image_vis_h, requires_grad=True).cuda()
            image_vis_scr,_,  R, L, _, _ = enhancemodel(image_vis, image_vis_h)  # 模型推理
            # image_vis_en = enh_module.enh(image_vis, R, L, image_vis_h_scr)
            # print(f"image_vis_scr shape: {image_vis_scr.shape}")  # 调试输出形状
            image_vis_en = image_vis_scr
            image_vis_en_ycbcr = rgb2ycbcr(image_vis_scr)
            image_ir = Variable(image_ir, requires_grad=True).float().cuda()
            vi_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
            _, _, _, Y_f = net(vi_en_y, image_ir)
            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)

            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)

            loss_fusion, loss_color, loss_grad,loss_image= criteria_fusion(vi_en_y, image_vis_en, image_ir, Y_f, I_f)

            ssim_loss = 0
            ssim_loss_temp = 1 - final_ssim(image_ir, vi_en_y, Y_f)
            ssim_loss += ssim_loss_temp
            ssim_loss /= len(Y_f)
            loss_fusion = loss_fusion + 30 * ssim_loss
            loss_fusion.backward()
            for name, param in net.named_parameters():
                if param.grad is None:
                    print(name, param.grad_fn)
            if grad_step > 1:
                loss_fusion = loss_fusion / grad_step
            if (it + 1) % grad_step == 0:
                optim.step()
                optim.zero_grad()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = dl.n_iter * epo + it + 1
            if (it + 1) % 20 == 0:
                lr = optim.param_groups[0]['lr']
                eta = int((dl.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(
                    ['step: {it}/{max_it}',
                     'loss_fusion:{loss_fusion:.4f}\n',
                     'loss_image: {loss_image:.4f}',
                     'loss_grad: {loss_grad:4f}',
                     'loss_color: {loss_color:4f}',
                     # 'loss_gc: {loss_gc:4f}',
                     'loss_ssim:{loss_ssim:4f}',
                     'eta: {eta}',
                     'time: {time:.4f}', ]).format(
                    it=now_it, max_it=dl.n_iter * epoch, lr=lr,
                    loss_fusion=loss_fusion, loss_image=loss_image,
                    loss_grad=loss_grad,
                    loss_color=loss_color,
                    loss_ssim=ssim_loss, eta=eta, time=t_intv, )
                logger.info(msg)
                st = ed

            # 保存最小损失模型
            if loss_fusion < best_loss:
                best_loss = loss_fusion
                best_epoch = epo + 1
                best_model = net.state_dict()  # 保存当前模型的权重

    # 保存最佳模型
    if best_model is not None:
        best_model_path = osp.join(modelpth, 'fusion_model_best.pth')
        torch.save(best_model, best_model_path)
        logger.info(f'Best model saved at epoch {best_epoch} with loss: {best_loss:.4f} to: {best_model_path}')

    # 保存当前模型（最后一次训练的模型）
    save_pth = osp.join(modelpth, 'fusion_model.pth')
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('Fusion Model Training done~, The Model is saved to: {}'.format(save_pth))
    logger.info('\n')


# def train_fusion(i, logger=None):  # 再增强与融合网络的训练函数
#     modelpth = './model'
#     modelpth = osp.join(modelpth, str(i + 1))
#     os.makedirs(modelpth, mode=0o777, exist_ok=True)
#     fusion_batch_size = 5
#     enhance_model_path = osp.join(os.getcwd(), 'model', str(i + 1), 'epoch_90.pth')
#     n_workers = 4
#     train_transforms = transforms.Compose([transforms.RandomCrop((128,128)),
#                                            transforms.ToTensor()])
#     ds = dataloader.fusion_dataset_loader('train',train_transforms)  # load training data
#     dl = torch.utils.data.DataLoader(ds, batch_size=fusion_batch_size, shuffle=True, num_workers=n_workers,
#                                      pin_memory=False)
#     # dl是增强之后的vi_en和ir
#     net = model.FusionNet()
#     if i == 0:
#         net.apply(weights_init)
#     if i > 0:  # 第二次训练加载上一次训练的字典
#         load_path = './model'
#         load_path = osp.join(load_path, str(i), 'fusion_model_best.pth')
#         net.load_state_dict(torch.load(load_path))
#         print('Load Pre-trained Fusion Model:{}!'.format(load_path))  # 加载之前训练的fusion模型
#     net.train()
#     enhancemodel = enhance_model.DecomNet().cuda()  # 亮度调整网络
#     enhancemodel.load_state_dict(torch.load(enhance_model_path))
#     enhancemodel.eval()
#     enh_module = enhance_model.EnhStage()
#     lr_start = 5e-4
#     optim = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=0.0001)
#     criteria_fusion = fusionloss()
#     st = glob_st = time.time()
#     epoch = 200
#     grad_step = 5.0  # gradient accumulation to increase batch_size
#     dl.n_iter = len(dl)
#     # 保存最小损失对应的模型
#     best_loss = float('inf')  # 初始时设置最小损失为无穷大
#     best_model = None  # 用来保存最小损失时的模型
#     best_epoch = 0  # 记录最佳模型所在的 epoch
#
#     for epo in range(0, epoch):
#         lr_decay = 0.75
#         lr_this_epo = lr_start * lr_decay ** ((epo / 5) + 1)  # 迭代变更初始学习率
#         for param_group in optim.param_groups:
#             param_group['lr'] = lr_this_epo
#         for it, (image_vis, image_ir, image_vis_h) in enumerate(dl):
#             net.train().cuda()
#             image_vis = Variable(image_vis, requires_grad=True).float().cuda()
#             image_vis_h = Variable(image_vis_h, requires_grad=True).cuda()
#             _, image_vis_h_scr, R, L, _, _ = enhancemodel(image_vis, image_vis_h)  # 模型推理
#             image_vis_en = enh_module.enh(image_vis, R, L, image_vis_h_scr)
#             image_vis_en_ycbcr = rgb2ycbcr(image_vis_en)
#             image_ir = Variable(image_ir, requires_grad=True).float().cuda()
#             vi_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
#             _, _, _, Y_f = net(vi_en_y, image_ir)
#             fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
#             I_f = ycbcr2rgb(fusion_ycbcr)
#
#             ones = torch.ones_like(I_f)
#             zeros = torch.zeros_like(I_f)
#             I_f = torch.where(I_f > ones, ones, I_f)
#             I_f = torch.where(I_f < zeros, zeros, I_f)
#
#             loss_fusion, loss_image, loss_grad, loss_color = criteria_fusion(vi_en_y, image_vis_en, image_ir, Y_f, I_f)
#
#             ssim_loss = 0
#             ssim_loss_temp = 1 - final_ssim(image_ir, vi_en_y, Y_f)
#             ssim_loss += ssim_loss_temp
#             ssim_loss /= len(Y_f)
#             loss_fusion = loss_fusion + 30 * ssim_loss
#             loss_fusion.backward()
#             torch.cuda.empty_cache()
#             for name, param in net.named_parameters():
#                 if param.grad is None:
#                     print(name, param.grad_fn)
#             if grad_step > 1:
#                 loss_fusion = loss_fusion / grad_step
#             if (it + 1) % grad_step == 0:
#                 optim.step()
#                 optim.zero_grad()
#             ed = time.time()
#             t_intv, glob_t_intv = ed - st, ed - glob_st
#             now_it = dl.n_iter * epo + it + 1
#             if (it + 1) % 50 == 0:
#                 lr = optim.param_groups[0]['lr']
#                 eta = int((dl.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
#                 eta = str(datetime.timedelta(seconds=eta))
#                 msg = ', '.join(
#                     ['step: {it}/{max_it}',
#                      'loss_fusion:{loss_fusion:.4f}\n',
#                      'loss_image: {loss_image:.4f}',
#                      'loss_grad: {loss_grad:4f}',
#                      'loss_color: {loss_color:4f}',
#                      'loss_ssim:{loss_ssim:4f}',
#                      'eta: {eta}',
#                      'time: {time:.4f}', ]).format(
#                     it=now_it, max_it=dl.n_iter * epoch, lr=lr,
#                     loss_fusion=loss_fusion, loss_image=loss_image,
#                     loss_grad=loss_grad, loss_color=loss_color,
#                     loss_ssim=ssim_loss, eta=eta, time=t_intv, )
#                 logger.info(msg)
#                 st = ed
#
#             # 保存最小损失模型
#             if loss_fusion < best_loss:
#                 best_loss = loss_fusion
#                 best_epoch = epo + 1
#                 best_model = net.state_dict()  # 保存当前模型的权重
#
#     # 保存最佳模型
#     if best_model is not None:
#         best_model_path = osp.join(modelpth, 'fusion_model_best.pth')
#         torch.save(best_model, best_model_path)
#         logger.info(f'Best model saved at epoch {best_epoch} with loss: {best_loss:.4f} to: {best_model_path}')
#
#     # 保存当前模型（最后一次训练的模型）
#     save_pth = osp.join(modelpth, 'fusion_model.pth')
#     state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#     torch.save(state, save_pth)
#     logger.info('Fusion Model Training done~, The Model is saved to: {}'.format(save_pth))
#     logger.info('\n')



def train_enhancement(num, logger=None):
    enhance_batch_size = 5
    n_workers = 8
    lr_start = 1e-4
    image_size = 256
    patch_size = 256
    modelpth = './model'
    modelpth = osp.join(modelpth, str(num + 1))
    enhancemodel = enhance_model.DecomNet().cuda()
    enhancemodel.apply(weights_init)
    enhancemodel.train()
    optimizer = torch.optim.Adam(enhancemodel.parameters(), lr=lr_start, weight_decay=0.0001)
    # load LFN
    if num > 0:
        fusionmodel = model.FusionNet()
        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        fusionmodel_path = osp.join(os.getcwd(), 'model', str(num), 'epoch_45.pth')
        fusionmodel.load_state_dict(torch.load(fusionmodel_path), False)
        fusionmodel = fusionmodel.cuda()
        fusionmodel.eval()
        for q in fusionmodel.parameters():
            q.requires_grad = False
        # Freeze LFN network parameters
    tr=transforms.Compose([transforms.RandomCrop(((patch_size, patch_size))),
                                           transforms.ToTensor()])
    datas = dataloader.enhance_dataset_loader(data_dir='train/vi',tr=tr,img_size=(image_size,image_size)) # load training data
    datal = torch.utils.data.DataLoader(datas, batch_size=enhance_batch_size, shuffle=True, num_workers=n_workers,
                                        pin_memory=False)
    print("the training dataset is length:{}".format(len(datas)))
    datal.n_iter = len(datal)

    # fusion_loss
    L_r =enhancement_loss.LRetinex()
    L_l = enhancement_loss.LIlluination()
    L_exp = enhancement_loss.LExp()
    L_sc = enhancement_loss.LSC()
    L_hue = enhancement_loss.LHue()

    grad_acc_steps = 4.0  # gradient accumulation to increase batch_size
    epoch = 20
    st = glob_st = time.time()
    logger.info('Training Enhancement Model start~')
    for epo in range(epoch):
        # lr_decay = 0.75
        # lr_this_epo = lr_start * (lr_decay ** ((epo / 5) + 1))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_this_epo

        for it, (input, input_h) in enumerate(datal):
            enhancemodel.train()
            input, input_h = input.cuda(), input_h.cuda()
            input_scr, input_h_scr, R, L, R_hue, input_hue = enhancemodel(input, input_h)

            # 损失计算
            l_r = L_r(input, R, L)
            l_l = L_l(input, L)
            l_exp = L_exp(input_scr)
            l_sc = L_sc(input_h_scr, R)
            l_hue = L_hue(input_hue, R_hue)
            loss_enhance = 60 * l_r + l_l + l_sc + 2 * l_exp + 2 * l_hue
            loss_total = loss_enhance / grad_acc_steps
            loss_total.backward()

            # 梯度累积
            if (it + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 日志输出
            if (it + 1) % 50 == 0:
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = len(datal) * epo + it + 1
                eta = str(datetime.timedelta(seconds=int((len(datal) * epoch - now_it) * (glob_t_intv / now_it))))
                logger.info(
                    f"Step {now_it}/{len(datal) * epoch}, loss_total: {loss_total.item():.4f}, eta: {eta}, time: {t_intv:.4f}")
                st = ed

    enhance_model_file = osp.join(modelpth, 'enhancement_model.pth')
    torch.save(enhancemodel.state_dict(), enhance_model_file)
    logger.info(f"Enhancement Model saved to: {enhance_model_file}")
    logger.info('\n')


def run_enhance(i):  # LAN eval
    # 定义模型和保存路径
    # enhance_model_path = osp.join(os.getcwd(), 'model', str(i + 1), 'epoch_90.pth')
    enhance_model_path = 'checkpoint/epoch_90.pth'
    enhanced_dir = osp.join(os.getcwd(), 'eval/vi_en', str(i + 1))
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)  # 确保子文件夹存在

    # 加载模型
    enhancemodel = enhance_model.DecomNet().cuda()
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    enhancemodel.eval()
    print('Enhancement model loaded successfully.')

    # 加载测试数据集
    val_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = dataloader.enhance_dataset_loader_test(data_dir='eval/vi',tr=val_transforms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # 可根据内存和性能调整
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )

    # 进度条
    test_tqdm = tqdm(test_loader, total=len(test_loader), desc="Running Enhancement Evaluation")
    enh_module = enhance_model.EnhStage()
    with torch.no_grad():
        for input, input_h, name in test_tqdm:
            input = input.cuda()
            input_h = input_h.cuda()

            input_scr, input_h_scr, R, L, R_hue, input_hue = enhancemodel(input, input_h)
            out = input_scr  # 使用增强的图像作为输出

            # 创建图像保存文件夹
            I_folder = os.path.join(enhanced_dir)  # 增强图像
            # R_folder = os.path.join(enhanced_dir, 'R')  # 反射图
            # L_folder = os.path.join(enhanced_dir, 'L')  # 亮度图

            # 如果文件夹不存在，创建它们
            os.makedirs(I_folder, exist_ok=True)
            # os.makedirs(R_folder, exist_ok=True)
            # os.makedirs(L_folder, exist_ok=True)

            for k in range(len(name)):
                # 获取文件名
                file_name = os.path.splitext(name[k])[0]  # 获取图像文件名，不包含扩展名

                # 将张量限制在[0, 1]范围内，避免保存时颜色溢出
                out_k = torch.clamp(out[k], 0, 1)
                R_k = torch.clamp(R[k], 0, 1)
                L_k = torch.clamp(L[k], 0, 1)

                # 使用 `save_image` 保存图像，直接保存张量
                save_image(out_k, os.path.join(I_folder, f'{file_name}.jpg'))  # 保存增强图像
                # save_image(R_k, os.path.join(R_folder, f'{file_name}_R.jpg'))  # 保存反射图
                # save_image(L_k, os.path.join(L_folder, f'{file_name}_L.jpg'))  # 保存亮度图
    # with torch.no_grad():
    #     for input, input_h, name in test_tqdm:
    #         input = input.cuda()
    #         input_h = input_h.cuda()
    #         input_scr, input_h_scr, R, L, R_hue, input_hue = enhancemodel(input, input_h)
    #         # out = enh_module.enh(input, R, L, input_h_scr)
    #         out = input_scr
    #         enhanced_image = out.cpu().numpy()  # 转为array
    #         for k in range(len(name)):
    #             image = enhanced_image[k, :, :, :]  # 丢掉第0维
    #             image = image.squeeze()
    #             image = torch.tensor(image).to(input.device)
    #             image = transforms.ToPILImage()(image)
    #
    #             # 设置保存路径并确保目录存在
    #             save_path = osp.join(enhanced_dir, name[k])
    #             os.makedirs(os.path.dirname(save_path), mode=0o777, exist_ok=True)
    #             image.save(save_path)


def run_fusion(i):  # RFN eval
    fusion_model_path = osp.join(os.getcwd(), 'model/CDFF', str(i + 1), 'fusion_model_best.pth')
    fusion_dir = osp.join(os.getcwd(), 'eval')
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
    fusionmodel = model.FusionNet().cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel, done!')

    testdataset = dataloader.fusion_dataset_loader_eval(i, osp.join(os.getcwd(), 'eval'))
    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    testtqdm = tqdm(testloader, total=len(testloader))
    with torch.no_grad():
        for images_vis, images_ir, name in testtqdm:
            images_vis, images_ir = images_vis.cuda(), images_ir.cuda()
            image_vis_en_ycbcr = rgb2ycbcr(images_vis)
            image_vis_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
            _, _, _, Y_f = fusionmodel(image_vis_en_y, images_ir)
            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)

            I_f = torch.clamp(I_f, 0, 1)  # 将 I_f 限制在 [0, 1] 范围
            I_f = I_f.cpu().numpy()
            for k in range(len(name)):
                image_I_f = I_f[k, :, :, :].squeeze()
                image_I_f = torch.tensor(image_I_f).to(images_vis.device)
                image_I_f = transforms.ToPILImage()(image_I_f)

                save_path1 = osp.join(fusion_dir, 'If', str(i + 1), name[k])
                os.makedirs(os.path.dirname(save_path1), mode=0o777, exist_ok=True)  # 确保目录存在
                image_I_f.save(save_path1)


# def run_enhance(i):  # LAN eval
#     # 定义模型和保存路径
#     enhance_model_path = osp.join(os.getcwd(), 'model', str(i + 1), 'epoch_90.pth')
#     enhanced_dir = osp.join(os.getcwd(), 'test/LLVIP/vi_en')
#     os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)  # 确保子文件夹存在
#
#     # 加载模型
#     enhancemodel = enhance_model.DecomNet().cuda()
#     enhancemodel.load_state_dict(torch.load(enhance_model_path))
#     enhancemodel.eval()
#     print('Enhancement model loaded successfully.')
#
#     # 加载测试数据集
#     val_transforms = transforms.Compose([transforms.ToTensor()])
#     test_dataset = dataloader.enhance_dataset_loader_test(data_dir='test/LLVIP/vi',tr=val_transforms)
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=1,  # 可根据内存和性能调整
#         shuffle=False,
#         num_workers=4,
#         pin_memory=False,
#         drop_last=False,
#     )
#
#     # 进度条
#     test_tqdm = tqdm(test_loader, total=len(test_loader), desc="Running Enhancement Evaluation")
#     enh_module = enhance_model.EnhStage()
#     with torch.no_grad():
#         for input, input_h, name in test_tqdm:
#             input = input.cuda()
#             input_h = input_h.cuda()
#             input_scr, input_h_scr, R, L, R_hue, input_hue = enhancemodel(input, input_h)
#             out = enh_module.enh(input, R, L, input_h_scr)
#             enhanced_image = out.cpu().numpy()  # 转为array
#             for k in range(len(name)):
#                 image = enhanced_image[k, :, :, :]  # 丢掉第0维
#                 image = image.squeeze()
#                 image = torch.tensor(image).to(input.device)
#                 image = transforms.ToPILImage()(image)
#
#                 # 设置保存路径并确保目录存在
#                 save_path = osp.join(enhanced_dir, name[k])
#                 os.makedirs(os.path.dirname(save_path), mode=0o777, exist_ok=True)
#                 image.save(save_path)
#
#
# def run_fusion(i):  # RFN eval
#     fusion_model_path = osp.join(os.getcwd(), 'model', str(i + 1), 'fusion_model_best.pth')
#     fusion_dir = osp.join(os.getcwd(), 'test')
#     os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
#     fusionmodel = model.FusionNet().cuda()
#     fusionmodel.eval()
#     fusionmodel.load_state_dict(torch.load(fusion_model_path))
#     print('fusionmodel, done!')
#
#     testdataset = dataloader.fusion_dataset_loader_test(osp.join(os.getcwd(), 'test/LLVIP/'))
#     testloader = DataLoader(
#         dataset=testdataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=False,
#         drop_last=False,
#     )
#     testtqdm = tqdm(testloader, total=len(testloader))
#     with torch.no_grad():
#         for images_vis, images_ir, name in testtqdm:
#             images_vis, images_ir = images_vis.cuda(), images_ir.cuda()
#
#             image_vis_en_ycbcr = rgb2ycbcr(images_vis)
#             image_vis_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
#             _, _, _, Y_f = fusionmodel(image_vis_en_y, images_ir)
#             fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
#             I_f = ycbcr2rgb(fusion_ycbcr)
#
#             I_f = torch.clamp(I_f, 0, 1)  # 将 I_f 限制在 [0, 1] 范围
#             I_f = I_f.cpu().numpy()
#             for k in range(len(name)):
#                 image_I_f = I_f[k, :, :, :].squeeze()
#                 image_I_f = torch.tensor(image_I_f).to(images_vis.device)
#                 image_I_f = transforms.ToPILImage()(image_I_f)
#
#                 save_path1 = osp.join(fusion_dir, 'LLVIP/If',  name[k])
#                 os.makedirs(os.path.dirname(save_path1), mode=0o777, exist_ok=True)  # 确保目录存在
#                 image_I_f.save(save_path1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(0, 10):
        # train_enhancement(i, logger)  # LAN train
        # print("|{0} Train LAN Sucessfully~!".format(i + 1))
        run_enhance(i)  # LAN eval
        print("|{0} Enhancement Image Sucessfully~!".format(i + 1))
        train_fusion(i, logger)  # RFN train
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion(i)  # RFN eval
        print("|啊哈哈|{0} Fusion Image Sucessfully~!".format(i + 1))
        print("———————————————————————————")
    print("Training Done!")
