import argparse
import torch
import sys
import os

from torch.optim import lr_scheduler
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import vgg_loss
from model import DecomNet
from DataLoader import LoadDataset, LoadDataset1
import enhanceloss
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import warnings

warnings.filterwarnings('ignore')


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()




def load_lossfn():
    L_r = enhanceloss.LRetinex()
    L_l = enhanceloss.LIlluination()
    L_exp = enhanceloss.LExp()
    L_sc = enhanceloss.LSC()
    L_hue = enhanceloss.LHue()

    return L_r, L_l, L_exp, L_sc, L_hue


def train(opt, device, train_loader, val_loader, model, optimizer,scheduler):
    """
    训练函数，执行模型的训练与验证过程，并记录训练日志和模型的保存。

    参数:
        opt: 配置选项，包含训练参数，如 epochs、ckpt_folder、save_period 等。
        device: 训练设备 (如 'cuda' 或 'cpu')。
        train_loader: 加载训练数据的 DataLoader。
        val_loader: 加载验证数据的 DataLoader。
        model: 需要训练的模型。
        optimizer: 优化器，用于更新模型参数。
    """
    # 是否加载权重的开关
    load_weights_flag = True  # 将这个变量设置为 False 来关闭加载权重

    # 加载权重的逻辑
    if load_weights_flag and opt.load_weights and os.path.exists(opt.load_weights):
        print(f"Loading weights from {opt.load_weights}")
        model.load_state_dict(torch.load(opt.load_weights, map_location=device))
    else:
        print("Not loading weights")

    # 创建用于保存训练图像和验证图像的文件夹
    train_img_folder = os.path.join(opt.ckpt_folder, 'train_images')
    val_img_folder = os.path.join(opt.ckpt_folder, 'val_images/vien')

    if not os.path.exists(train_img_folder):
        os.makedirs(train_img_folder)
    if not os.path.exists(val_img_folder):
        os.makedirs(val_img_folder)
    # 创建 TensorBoard 日志记录器 (如果开启日志保存选项)
    summary_writer = SummaryWriter(os.path.join(opt.ckpt_folder, 'logs')) if opt.save_graph else ''

    # 加载损失函数，包含多个不同的损失项 (如 L_r, L_l 等)
    L_r, L_l, L_exp, L_sc, L_hue= load_lossfn()

    # 开始 epoch 训练
    for epoch in range(opt.epochs):
        print('\n||||===================== epoch {num} =====================||||'.format(num=epoch + 1))
        epoch_loss = []  # 用于存储每个 batch 的损失
        model.train().cuda()  # 切换模型到训练模式 (启用 dropout、batchnorm 等)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # 遍历训练集
        for _, (input, input_h) in enumerate(tqdm(train_loader, desc='Training', file=sys.stdout)):
            input = input.to(device)  # 将输入数据移动到指定设备 (如 GPU)
            input_h = input_h.to(device)  # 同样将高光图像也移动到设备上

            # 模型前向传播，得到模型的多个输出 (R, L, 等)
            input_scr, input_h_scr, R, L, R_hue, input_hue = model(input, input_h)

            # 计算不同损失
            l_r = L_r(input, R, L)  # 重建损失
            l_l = L_l(input, L)  # 亮度损失
            l_exp = L_exp(input_scr)  # 曝光损失
            l_sc = L_sc(input_h_scr, R)  # 色彩损失
            l_hue = L_hue(input_hue, R_hue)  # 色相损失

            # 综合多个损失项，生成总损失函数
            loss = 80 * l_r + 0.5 * l_l + 1.5 * l_sc + 1 * l_exp + 1 * l_hue

            epoch_loss.append(loss.item())  # 将当前 batch 的损失保存

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip_norm)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()  # 更新模型参数

        # 输出当前 epoch 的平均训练损失
        print('Train loss: ', sum(epoch_loss) / len(epoch_loss))

        # 如果开启了日志保存，记录训练损失到 TensorBoard
        if opt.save_graph:
            summary_writer.add_scalar('train_set', sum(epoch_loss) / len(epoch_loss), epoch)

        # 保存模型的检查点文件，每隔 opt.save_period 次 epoch 保存一次
        if ((epoch + 1) % opt.save_period) == 0:
            torch.save(model.state_dict(),
                       os.path.join(opt.ckpt_folder, 'epoch_{num}.pth'.format(num=epoch + 1)))
            # 生成并保存训练结果图像 (仅第一个 batch 的图像作为示例)
            save_image(input_scr,os.path.join(train_img_folder, 'train_epoch_{num}_vi_en.png'.format(num=epoch + 1)))
            save_image(R, os.path.join(train_img_folder, 'train_epoch_{num}_R.png'.format(num=epoch + 1)))
            save_image(L, os.path.join(train_img_folder, 'train_epoch_{num}_L.png'.format(num=epoch + 1)))

        # 每隔 opt.val_period 次 epoch 进行一次验证集的验证
        if ((epoch + 1) % opt.val_period) == 0:
            epoch_loss = []  # 清空损失列表，用于记录验证集损失
            with torch.no_grad():  # 验证阶段不需要计算梯度
                for idx, (input, input_h, img_name)in enumerate(tqdm(val_loader, desc='Validating', file=sys.stdout)):
                    input = input.to(device)  # 将输入移动到设备
                    input_clahe = input_h.to(device)  # CLAHE 预处理的图像也移动到设备

                    # 模型前向传播，得到模型的输出
                    input_scr, input_h_scr, R, L, R_hue, input_hue = model(input, input_clahe)

                    # 计算验证集上的各项损失
                    l_r = L_r(input, R, L)
                    l_l = L_l(input, L)
                    l_exp = L_exp(input_scr)
                    l_sc = L_sc(input_h_scr, R)
                    l_hue = L_hue(input_hue, R_hue)

                    # 计算总损失并记录
                    loss = 80 * l_r + 0.5 * l_l + 1.5 * l_sc + 1 * l_exp + 1 * l_hue

                    epoch_loss.append(loss.item())
                    # 保存验证图像 (每次验证都保存)
                    file_name = os.path.splitext(img_name[0])[0]  # 获取图像文件名，不包含扩展名
                    save_image(input_scr,os.path.join(val_img_folder, 'train_epoch_{num}_vi_en.png'.format(num=epoch + 1)))
                    # save_image(R, os.path.join(val_img_folder, 'val_epoch_{num}_R.jpg'.format(num=epoch + 1)))
                    # save_image(L, os.path.join(val_img_folder, 'val_epoch_{num}_L.jpg'.format(num=epoch + 1)))
                # 输出验证集的平均损失
                val_loss = sum(epoch_loss) / len(epoch_loss)
                print(f"Val loss: {val_loss:.6f}")


                # 记录验证集的损失到 TensorBoard (如果开启了日志记录)
                if opt.save_graph:
                    summary_writer.add_scalar('eval', sum(epoch_loss) / len(epoch_loss), epoch)
                    # 调整学习率
        # scheduler.step()
    # 关闭 TensorBoard 日志记录器
    if opt.save_graph:
        summary_writer.close()


def main(opt):
    # check input dir

    assert os.path.exists(opt.train_data), 'train_data folder {dir} does not exist'.format(dir=opt.train_data)
    assert os.path.exists(opt.val_data), 'val_data folder {dir} does not exist'.format(dir=opt.val_data)
    if not os.path.exists(opt.ckpt_folder):
        os.makedirs(opt.ckpt_folder)
    device_id = 'cuda:' + opt.device
    device = torch.device(device_id if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')

    # load datasets
    train_transforms = transforms.Compose([transforms.RandomCrop((opt.patch_size, opt.patch_size)),
                                           transforms.ToTensor()])
    train_set = LoadDataset(opt.train_data, train_transforms, img_size=(opt.imgsz, opt.imgsz))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers,
                              batch_size=opt.batch_size, shuffle=True)
    val_transforms = transforms.Compose([transforms.ToTensor()])
    eval = LoadDataset1(opt.val_data, val_transforms, img_size=(1280,1024))
    val_loader = DataLoader(dataset=eval, num_workers=opt.num_workers,
                            batch_size=1, shuffle=True)

    print('Num of train_set set: {num}'.format(num=len(train_set)))
    print('Num of val_set set: {num}'.format(num=len(eval)))

    model = DecomNet().to(device)
    model.apply(init_weights)

    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    # 设置余弦退火学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # strat training
    train(opt, device, train_loader, val_loader, model, optimizer,scheduler)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='ours_dataset_240/train_set/vi', help='train_set set path')
    parser.add_argument('--val_data', type=str, default='ours_dataset_240/eval/vi', help='val_set set path')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for training')
    parser.add_argument('--patch_size', type=int, default=64, help='random crop imgsz during training')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    parser.add_argument('--imgsz', type=int, default=256, help='input image size')
    parser.add_argument('--epochs', type=int, default=500, help='total epochs')
    parser.add_argument('--lr', type=float, default=1e-05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--device', default='0', help='use cuda device; 0, 1, 2 or cpu')
    parser.add_argument('--val_period', type=int, default=500, help='perform validation every x epoch')
    parser.add_argument('--save_period', type=int, default=10, help='save checkpoint every x epoch')
    parser.add_argument('--ckpt_folder', type=str, default='ckpts/exp1/new_2', help='location for saving ckpts')
    parser.add_argument('--save_graph', action='store_true', default=True,
                        help='generate graph of training updating process')
    parser.add_argument('--load_weights', type=str, default='ckpts/exp1/new_2/epoch_500.pth', help='Path to pretrained weights')  #ckpts/exp1/epoch_55.pth
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
