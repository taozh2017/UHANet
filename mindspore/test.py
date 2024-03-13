import pdb

import mindspore.numpy as np
import os, argparse
import mindspore as ms
from mindspore import ops, nn
from Src.CODNet import PolypNet
from Src.utils.dataloader import get_loader, get_loader_test

import cv2

ms.set_context(device_target='GPU')

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./Snap/CFANet.pth')

for _data_name in ['BUSI']:  # GLAS

    print('-----------starting -------------')

    data_path = '../data/{}/test/'.format(_data_name)
    save_path = './Snapshot/seg_maps/{}/'.format(_data_name)

    opt = parser.parse_args()
    model = PolypNet(channel=64)

    param_dict = ms.load_checkpoint(opt.pth_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)

    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    test_loader = get_loader_test(image_root, gt_root, testsize=opt.testsize)

    for i, pack in enumerate(test_loader.create_tuple_iterator()):
        print(['--------------processing-------------', i])

        image, gt, name = pack
        image = ops.ExpandDims()(image, 0)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        _, _, _, _, res = model(image)

        res = ops.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid()[0]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + str(name.asnumpy()), ops.Transpose()(res, (1, 2, 0)).asnumpy() * 255)
