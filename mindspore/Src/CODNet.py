import pdb

from mindspore import Tensor, ops, Parameter, nn, common
import mindspore as ms
from Src.res2net_v1b_base import Res2Net_model


class global_module(nn.Cell):
    def __init__(self, channels=64, r=4):
        super(global_module, self).__init__()
        out_channels = int(channels // r)
        # local_att

        self.global_att = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(out_channels, use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(channels, use_batch_statistics=True)
        )

        self.sig = nn.Sigmoid()

    def construct(self, x):
        xg = self.global_att(x)
        out = self.sig(xg)

        return out


class BasicConv2d(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, pad_mode='valid', dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, dilation=dilation, has_bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention_avg(nn.Cell):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_avg, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc12 = nn.Conv2d(in_planes, in_planes // ratio, 1, stride=1, pad_mode='valid', has_bias=False)
        self.fc22 = nn.Conv2d(in_planes // ratio, in_planes, 1, stride=1, pad_mode='valid', has_bias=False)

    def construct(self, x):
        x_avg_1 = self.avg_pool(x)
        x_avg_2 = self.fc12(x_avg_1)
        x_avg_out = self.fc22(self.relu(x_avg_2))

        out = x_avg_out

        return self.sigmoid(out)


class ChannelAttention_max(nn.Cell):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_max, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc11 = nn.Conv2d(in_planes, in_planes // ratio, 1, stride=1, pad_mode='valid', has_bias=False)
        self.fc21 = nn.Conv2d(in_planes // ratio, in_planes, 1, stride=1, pad_mode='valid', has_bias=False)

    def construct(self, x):
        x_max_1 = self.max_pool(x)
        x_max_2 = self.fc11(x_max_1)
        x_max_out = self.fc21(self.relu(x_max_2))

        out = x_max_out

        return self.sigmoid(out)


class eca_layer(nn.Cell):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, pad_mode='same', has_bias=False)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).swapaxes(-1, -2)).swapaxes(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SpatialAttention(nn.Cell):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, stride=1, pad_mode='same', has_bias=False)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        max_out, _ = x.max(axis=1, keepdims=True, return_indices=True)
        x = max_out
        x = self.conv1(x)

        return self.sigmoid(x)


class U_fusion_IM3(nn.Cell):
    def __init__(self, in_channel1, in_channel2, out_channel):
        self.init__ = super(U_fusion_IM3, self).__init__()

        act_fn = nn.ReLU()

        self.layer_rec1 = nn.SequentialCell(
            nn.Conv2d(in_channel1, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel), act_fn)
        self.layer_rec2 = nn.SequentialCell(
            nn.Conv2d(in_channel2, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel), act_fn)

        self.layer_uncer_1 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, pad_mode='same', has_bias=True), )
        self.layer_uncer_2 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, pad_mode='same', has_bias=True), )

        self.layer_cat = nn.SequentialCell(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel), act_fn)

    def construct(self, x1, x2):
        ######################################################
        x_rec1 = self.layer_rec1(x1)
        x_rec2 = self.layer_rec2(x2)

        x_in1 = x_rec1.unsqueeze(1)
        x_in2 = x_rec2.unsqueeze(1)

        x_cat = ops.concat((x_in1, x_in2), axis=1)
        x_max = x_cat.max(axis=1)
        x_max1 = x_max

        uncert_out1 = self.layer_uncer_1(x_max1)
        uncert_out1 = nn.Softmax(axis=1)(uncert_out1)

        x_top1 = uncert_out1 * ops.log(uncert_out1 + 1e-15)
        x_bottom1 = ops.log(Tensor([uncert_out1.shape[1]], ms.float32))
        uncert_1 = (- x_top1 / x_bottom1).sum(axis=1)
        x_max_att = 1 - uncert_1

        ###
        x_mul = x_rec1.mul(x_rec2)
        x_mul1 = x_mul

        uncert_out2 = self.layer_uncer_2(x_mul1)
        uncert_out2 = nn.Softmax(axis=1)(uncert_out2)

        x_top2 = uncert_out2 * ops.log(uncert_out2 + 1e-15)
        x_bottom2 = ops.log(Tensor([uncert_out2.shape[1]], ms.float32))
        uncert_2 = (- x_top2 / x_bottom2).sum(axis=1)
        x_mul_att = 1 - uncert_2

        x_max_att = x_max * x_max_att.unsqueeze(1)
        x_mul_att = x_mul * x_mul_att.unsqueeze(1)

        out = self.layer_cat(ops.concat((x_max_att, x_mul_att), axis=1))

        return out


class fea_aggre(nn.Cell):
    def __init__(self, in_channel, out_channel):
        self.init__ = super(fea_aggre, self).__init__()

        self.act_fn = nn.ReLU()

        self.layer_conv1 = BasicConv2d(in_channel, out_channel, 1)
        self.layer_conv2 = BasicConv2d(in_channel, out_channel, 1)
        self.layer_conv3 = BasicConv2d(in_channel, out_channel, 1)
        self.layer_conv4 = BasicConv2d(in_channel, out_channel, 1)

        self.layer_dil1 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True, dilation=1),
            nn.BatchNorm2d(out_channel), self.act_fn)
        self.layer_dil2 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True, dilation=2),
            nn.BatchNorm2d(out_channel), self.act_fn)
        self.layer_dil3 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True, dilation=5),
            nn.BatchNorm2d(out_channel), self.act_fn)

        self.layer_cat = nn.SequentialCell(
            nn.Conv2d(out_channel * 3, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel), self.act_fn)
        self.layer_out = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel * 2, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel * 2), self.act_fn)

    def construct(self, x):
        ######################################################

        x1 = self.layer_conv1(x)
        x2 = self.layer_conv2(x)
        x3 = self.layer_conv3(x)
        x4 = self.layer_conv4(x)

        x_dil3 = self.layer_dil3(x3)
        x_dil2 = self.layer_dil2(x2 + x_dil3)
        x_dil1 = self.layer_dil1(x1 + x_dil2)

        x_cat = self.layer_cat(ops.concat((x_dil3, x_dil2, x_dil1), axis=1))

        out = self.layer_out(x_cat + x4)

        return out


class FAM(nn.Cell):
    def __init__(self, in_channel):
        super(FAM, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layer_rec1 = nn.SequentialCell(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(32), self.relu)
        self.layer_rec2 = nn.SequentialCell(
            nn.Conv2d(1024, 32, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(32), self.relu)
        self.layer_rec3 = nn.SequentialCell(
            nn.Conv2d(2048, 32, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(32), self.relu)

        self.layer_gl_1 = nn.SequentialCell(nn.Conv2d(32, 32, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
                                            nn.BatchNorm2d(32),
                                            self.relu)
        self.layer_gl_2 = nn.SequentialCell(nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
                                            nn.BatchNorm2d(64),
                                            self.relu)

        self.atten_ch1 = ChannelAttention_max(32)
        self.atten_ch2 = ChannelAttention_max(32)

        ## ---------------------------------------- ##

        self.down_2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x1, x2, x3):
        b, c, h, w = x3.shape
        x1_1 = self.down_2(self.layer_rec1(x1))
        x1_2 = self.layer_rec2(x2)
        x1_3 = ops.interpolate(self.layer_rec3(x3), size=(h * 2, w * 2), mode='bilinear', align_corners=True)

        x12 = x1_1 + x1_2
        x13 = self.layer_gl_1((x12.mul(self.atten_ch1(x12)) + x1_2))

        x22 = x13 + x1_3
        x23 = self.layer_gl_2((x22.mul(self.atten_ch2(x22)) + x1_3))

        return x23


###############################################################################
###############################################################################
class PolypNet(nn.Cell):
    # resnet based encoder decoder
    def __init__(self, channel=64, opt=None):
        super(PolypNet, self).__init__()

        act_fn = nn.ReLU()

        self.resnet = Res2Net_model(50)
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)

        ## ---------------------------------------- ##

        ######################################################################

        self.fu_layer4 = U_fusion_IM3(1024, 2048, channel)
        self.fu_layer3 = U_fusion_IM3(512, 1024, channel)
        self.fu_layer2 = U_fusion_IM3(256, 512, channel)
        self.fu_layer1 = U_fusion_IM3(64, 256, channel)

        self.agg4 = fea_aggre(channel, channel // 2)
        self.agg3 = fea_aggre(channel, channel // 2)
        self.agg2 = fea_aggre(channel, channel // 2)
        self.agg1 = fea_aggre(channel, channel // 2)

        self.FAM = FAM(channel // 2)

        ###### global map

        self.layer_rec2 = nn.SequentialCell(
            nn.Conv2d(512, channel // 2, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))
        self.layer_rec3 = nn.SequentialCell(
            nn.Conv2d(1024, channel // 2, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))
        self.layer_rec4 = nn.SequentialCell(
            nn.Conv2d(2048, channel // 2, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))

        self.layer_gl1 = nn.SequentialCell(
            nn.Conv2d((channel // 2) * 3, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_gl2 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel), act_fn)
        self.sal_gl = nn.SequentialCell(nn.Conv2d(channel, 1, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))

        self.atten_channel = ChannelAttention_avg(channel)
        self.atten_spatial = SpatialAttention()

        self.layer_gl_2 = nn.SequentialCell(
            nn.Conv2d(512, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel), act_fn, nn.MaxPool2d(2, stride=4))
        self.layer_gl_3 = nn.SequentialCell(
            nn.Conv2d(1024, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel), act_fn, nn.MaxPool2d(2, stride=2))
        self.layer_gl_4 = nn.SequentialCell(
            nn.Conv2d(2048, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel), act_fn)

        self.sal_out4 = nn.SequentialCell(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))
        self.sal_out3 = nn.SequentialCell(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))
        self.sal_out2 = nn.SequentialCell(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))
        self.sal_out1 = nn.SequentialCell(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, pad_mode='valid', has_bias=True))

    def construct(self, xx):
        # ---- feature abstraction -----
        b, c, h, w = xx.shape
        x0, x1, x2, x3, x4 = self.resnet(xx)  # h/4,h/4,h/8,h/16,h/32

        #######################################################################
        ###### global map
        gl_out = self.FAM(x2, x3, x4)
        gl_sal = self.sal_gl(gl_out)  # h/16

        ### 4 layer
        fu_out41 = self.fu_layer4(x3, ops.interpolate(x4, size=(h // 16, w // 16), mode='bilinear', align_corners=True))
        fu_out42 = fu_out41 + gl_out  # h/16
        fu_out43 = self.agg4(fu_out42)
        sal_out4 = self.sal_out4(fu_out43)

        ### 3 layer
        fu_out31 = self.fu_layer3(x2, ops.interpolate(x3, size=(h // 8, w // 8), mode='bilinear', align_corners=True))
        fu_out32 = fu_out31 + ops.interpolate(fu_out43, size=(h // 8, w // 8), mode='bilinear', align_corners=True)
        fu_out33 = self.agg3(fu_out32)
        sal_out3 = self.sal_out3(fu_out33)

        ### 2 layer
        fu_out21 = self.fu_layer2(x1, ops.interpolate(x2, size=(h // 4, w // 4), mode='bilinear', align_corners=True))
        fu_out22 = fu_out21 + ops.interpolate(fu_out33, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        fu_out23 = self.agg2(fu_out22)
        sal_out2 = self.sal_out2(fu_out23)

        ### 1 layer
        fu_out11 = self.fu_layer1(ops.interpolate(x0, size=(h // 2, w // 2), mode='bilinear', align_corners=True),
                                  ops.interpolate(x1, size=(h // 2, w // 2), mode='bilinear', align_corners=True))
        fu_out12 = fu_out11 + ops.interpolate(fu_out23, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fu_out13 = self.agg1(fu_out12)
        sal_out1 = self.sal_out1(fu_out13)

        ### final prediction
        pred_sal5 = ops.interpolate(gl_sal, size=(h, w), mode='bilinear', align_corners=True)
        pred_sal4 = ops.interpolate(sal_out4, size=(h, w), mode='bilinear', align_corners=True)
        pred_sal3 = ops.interpolate(sal_out3, size=(h, w), mode='bilinear', align_corners=True)
        pred_sal2 = ops.interpolate(sal_out2, size=(h, w), mode='bilinear', align_corners=True)
        pred_sal1 = ops.interpolate(sal_out1, size=(h, w), mode='bilinear', align_corners=True)

        return pred_sal5, pred_sal4, pred_sal3, pred_sal2, pred_sal1
