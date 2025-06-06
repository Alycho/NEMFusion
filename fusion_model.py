import math
import torch.nn as nn
import basicblock as B
import torch
import math
import torch.nn.functional as F
import torchvision

import profile

import dataloader
from dataloader import device
from dual_attention_fusion_module import attention_fusion_weight
import torch.nn.init as init

'''
# ===================
# srresnet
# ===================
'''
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # print(f"Initializing Conv2d layer with shape {m.weight.shape}")
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        # print(f"Initializing ConvTranspose2d layer with shape {m.weight.shape}")
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):  ####16
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=15, stride=1, padding=7)  ####kernel_size=7
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print(out.size())
        out = self.spatial_attention(out) * out
        return out


"""
features = torch.rand((8, 64, 192, 192))
attention = CBAM(64)
result = attention(features)

print(result.size()) 
"""


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # groups = in_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(CBAM(in_channels))  # ASPPPooling(in_channels, out_channels)selfAttention(64,192*192,192*192)
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )
        # nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


"""
aspp = ASPP(64,[2,4,8])
x = torch.rand(2,64,192,192)
print(aspp(x).shape)

"""


class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        # modules = []
        # modules.append(nn.Sequential(
        # nn.Conv2d(in_channels, out_channels, 1, bias=False),
        # nn.BatchNorm2d(out_channels),
        # nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            # groups = in_channels , bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )
        # nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0 + x)
        y2 = self.layer3(y1 + x)
        y3 = self.layer4(y2 + x)
        # res = torch.cat([y0,y1,y2,y3], dim=1)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight, y1_weight, y2_weight, y3_weight], 2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        y3_weight = torch.unsqueeze(weight[:, :, 3], 2)
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 + y3_weight * y3
        return self.project(x_att + x)

    # aspp = MFEblock(64,[2,4,8])


# x = torch.rand(2,64,192,192)
# print(aspp(x).shape)


class SRResNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=3, nc=64, nb=3):
        super(SRResNet, self).__init__()

        # 保留特征提取的部分
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [MFEblock(nc, [2, 4, 8]) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        # 不再需要上采样部分
        m_tail = B.conv(nc, out_nc, mode='C')

        self.backbone = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)))
        self.tail = m_tail

        # 初始化权重
        self.apply(self._initialize_weights)
        # print("Tail weights after initialization:", self.tail.weight)
        # print("Tail biases after initialization:", self.tail.bias)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # 使用 He 初始化（Kaiming 初始化）
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)  # 将偏置初始化为0

    def forward(self, x):
        feature = self.backbone(x)
        return self.tail(feature)







class CSFblock(nn.Module):
    ###联合网络
    def __init__(self, in_channels, channels_1):
        super().__init__()

        # 用于特征调整而非上采样
        self.layer1 = nn.Sequential(
            oneConv(in_channels, channels_1, 1, 0, 1),
            oneConv(channels_1, in_channels, 1, 0, 1),
        )

        # 注意力机制
        self.Fgp = nn.AdaptiveAvgPool2d(1)
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_h, x_l):
        # 使用卷积而非插值进行特征调整
        x1 = x_h
        x2 = x_l  # 保留原始分辨率的特征
        if x1.size() != x2.size():
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # 特征融合
        x_f = x1 + x2
        Fgp = self.Fgp(x_f)

        # 注意力权重生成与融合
        x_se = self.layer1(Fgp)
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        x_se = torch.cat([x_se1, x_se2], 2)
        x_se = self.softmax(x_se)

        att_3 = torch.unsqueeze(x_se[:, :, 0], 2)
        att_5 = torch.unsqueeze(x_se[:, :, 1], 2)
        x1 = att_3 * x1
        x2 = att_5 * x2

        # 最终融合输出
        x_all = x1 + x2
        return x_all


"""
x = torch.rand(1,32,100,100) 
x1 = torch.rand(1,32,200,200)
net = CSFblock(32,16,2)
print(net)
y = net(x1,x)     
print(y.size())

"""


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.CSF1 = CSFblock(64, 32)  # 用于逐层特征融合
        self.CSF2 = CSFblock(64, 32)  # 用于逐层特征融合

    def forward(self, x):
        # 提取多尺度特征
        x1 = self.maxpool(x)  # 第一层特征
        x2 = self.maxpool(x1)  # 第二层特征
        x3 = self.maxpool(x2)  # 第三层特征

        # 逐层融合特征
        x2 = self.CSF1(x2, x3)  # 将第2层特征与第3层特征融合
        x1 = self.CSF2(x1, x2)  # 将第1层特征与融合后的第2层特征融合

        # 输出最终融合的特征图
        return x1  # 返回多尺度特征融合后的结果

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.Lrelu = nn.LeakyReLU()
        filter_n = 32
        self.de_conv1 = nn.Conv2d(3, filter_n * 4, 3, 1, 1, bias=True)
        self.de_conv2 = nn.Conv2d(filter_n * 4, filter_n * 2, 3, 1, 1, bias=True)
        self.de_conv3 = nn.Conv2d(filter_n * 2, filter_n, 3, 1, 1, bias=True)
        self.de_conv4 = nn.Conv2d(filter_n, 1, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(filter_n * 4)
        self.bn2 = nn.BatchNorm2d(filter_n * 2)
        self.bn3 = nn.BatchNorm2d(filter_n)
        self.rgb2ycbcr = dataloader.rgb2ycbcr
        self.ycbcr2rgb = dataloader.ycbcr2rgb

    def forward(self, feature):
        feature = self.Lrelu(self.bn1(self.de_conv1(feature)))
        feature = self.Lrelu(self.bn2(self.de_conv2(feature)))
        feature = self.Lrelu(self.bn3(self.de_conv3(feature)))
        Y_f = torch.tanh(self.de_conv4(feature))
        return Y_f


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.feature_extractor = SRResNet()  # 特征提取网络
        self.fusion = FPN()  # 特征融合网络
        self.decoder = decoder().cuda()

        # Initialize weights
        self.apply(initialize_weights)

    def forward(self, vi_clahe_y, ir):
        assert vi_clahe_y.shape[2:] == ir.shape[2:], f"Mismatch in input sizes: vi_clahe_y {vi_clahe_y.shape}, ir {ir.shape}"
        feature_vi_en = self.feature_extractor(vi_clahe_y)
        feature_ir = self.feature_extractor(ir)

        # 特征融合
        feature_y_f = attention_fusion_weight(feature_vi_en, feature_ir)
        Y_f = self.decoder(feature_y_f)

        save_ir = self.decoder(feature_ir)
        save_vi_en = self.decoder(feature_vi_en)
        save_y_f = self.decoder(feature_y_f)

        return save_ir, save_vi_en, save_y_f, Y_f

# 使用示例
# vi_clahe_y 和 ir 是输入的可见光和红外图像
# fusion_net = FusionNet()
# save_ir, save_y_f, save_vi_en, Y_f = fusion_net(vi_clahe_y, ir)


# from ptflops import get_model_complexity_info
# if __name__ == '__main__':
# model = SRResNet()
# flops, params = get_model_complexity_info(model, (3, 192, 192), as_strings=True, print_per_layer_stat=True)
# print('Flops:  ' + flops)
# print('Params: ' + params)