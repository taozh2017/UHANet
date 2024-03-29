import os
import argparse
import pdb

import numpy as np
import logging
from datetime import datetime

import mindspore as ms
from mindspore import context
from mindspore import nn, ops

from Src.CODNet import PolypNet
from Src.utils.dataloader import get_loader, get_loader_test

ms.set_context(device_target='GPU')  # , mode=context.GRAPH_MODE


class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, opt):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.opt = opt
        self.total_step = self.train_loader.get_dataset_size()

        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.size_rates = [0.75, 1, 1.25]
        self.eval_loss_func = nn.L1Loss()
        self.best_mae = 1
        self.best_epoch = 0
        self.decay_rate = 0.1
        self.decay_epoch = 30

    def forward_fn(self, images, gts):
        sal_out1, sal_out2, sal_out3, sal_out4, sal_out5 = self.model(images)
        loss_sal1 = self.structure_loss(sal_out1, gts)
        loss_sal2 = self.structure_loss(sal_out2, gts)
        loss_sal3 = self.structure_loss(sal_out3, gts)
        loss_sal4 = self.structure_loss(sal_out4, gts)
        loss_sal5 = self.structure_loss(sal_out5, gts)

        loss_total = loss_sal1 + loss_sal2 + loss_sal3 + loss_sal4 + loss_sal5
        return loss_total, loss_sal1, loss_sal2, loss_sal3, loss_sal4, loss_sal5

    def train_step(self, images, gts):
        (loss, loss_sal1, loss_sal2, loss_sal3, loss_sal4, loss_sal5), grads = self.grad_fn(images, gts)
        self.optimizer(grads)
        return loss, loss_sal1, loss_sal2, loss_sal3, loss_sal4, loss_sal5

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.model.set_train(True)
            self.adjust_lr(epoch)
            for step, data_pack in enumerate(self.train_loader.create_tuple_iterator(), start=1):
                images, gts = data_pack
                for rate in self.size_rates:
                    # ---- rescale ----
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    if rate != 1:
                        images = ops.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gts = ops.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    loss, loss_sal1, loss_sal2, loss_sal3, loss_sal4, loss_sal5 = self.train_step(images, gts)

                # -- output loss -- #
                if step % 10 == 0 or step == self.total_step:
                    print(
                        '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_sal1: {:0.4f} Loss_sal2: {:0.4f} Loss_sal3: {:0.4f} Loss_sal4: {:0.4f} Loss_sal5: {:0.4f} Loss_total: {:0.4f}]'.
                        format(datetime.now(), epoch, epochs, step, self.total_step, loss_sal1.asnumpy(),
                               loss_sal2.asnumpy(), loss_sal3.asnumpy(), loss_sal4.asnumpy(), loss_sal5.asnumpy(),
                               loss.asnumpy()))

                    logging.info(
                        '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_sal1: {:0.4f} Loss_sal2: {:0.4f} Loss_sal3: {:0.4f} Loss_sal4: {:0.4f} Loss_sal5: {:0.4f} Loss_total: {:0.4f}'.
                        format(epoch, opt.epoch, step, self.total_step, loss_sal1.asnumpy(),
                               loss_sal2.asnumpy(), loss_sal3.asnumpy(), loss_sal4.asnumpy(), loss_sal5.asnumpy(),
                               loss.asnumpy()))
            self.test(epoch)

            if epoch % self.opt.save_epoch == 0:
                ms.save_checkpoint(model, os.path.join(save_path, 'CODNet_%d.pth' % (epoch)))

    def test(self, epoch):
        self.model.set_train(False)
        mae_sum = 0
        for i, pack in enumerate(self.test_loader.create_tuple_iterator(), start=1):
            # ---- data prepare ----
            image, gt, name = pack
            image = ops.ExpandDims()(image, 0)

            # ---- forward ----
            _, _, _, _, res = self.model(image)
            res = ops.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = nn.Sigmoid()(res[0][0])
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += self.eval_loss_func(res, gt)

            # ---- recording loss ----
        mae = mae_sum / self.test_loader.get_dataset_size()
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, self.best_mae, self.best_epoch))
        if epoch == 1:
            self.best_mae = mae
            self.best_epoch = epoch
        else:
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_epoch = epoch

                ms.save_checkpoint(model, os.path.join(save_path, 'Cod_best.ckpt'))
                print('best epoch:{}'.format(epoch))

    def structure_loss(self, pred, mask):
        pred = nn.Sigmoid()(pred)
        weit = 1 + 5 * ops.Abs()(ops.AvgPool(kernel_size=31, strides=1, pad_mode='same')(mask) - mask)
        wbce = nn.BCELoss(reduction='none')(pred, mask)
        wbce = (weit * wbce).sum(axis=(2, 3)) / weit.sum(axis=(2, 3))

        inter = ((pred * mask) * weit).sum(axis=(2, 3))
        union = ((pred + mask) * weit).sum(axis=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

    def adjust_lr(self, epoch):
        decay = self.decay_rate ** (epoch // self.decay_epoch)
        self.optimizer.get_lr().set_data(self.opt.lr * decay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=5, help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/BUSI/')

    parser.add_argument('--train_img_dir', type=str, default='../data/BUSI/train/images/')
    parser.add_argument('--train_gt_dir', type=str, default='../data/BUSI/train/masks/')
    parser.add_argument('--train_eg_dir', type=str, default='../data/BUSI/train/edges/')

    parser.add_argument('--test_img_dir', type=str, default='../data/BUSI/test/images/')
    parser.add_argument('--test_gt_dir', type=str, default='../data/BUSI/test/masks/')
    parser.add_argument('--test_eg_dir', type=str, default='../data/BUSI/test/edges/')

    opt = parser.parse_args()

    ms.set_context(device_id=opt.gpu)

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=opt.save_model + '/log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("COD-Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(opt.epoch,
                                                                                                            opt.lr,
                                                                                                            opt.batchsize,
                                                                                                            opt.trainsize,
                                                                                                            opt.clip,
                                                                                                            opt.decay_rate,
                                                                                                            opt.save_model,
                                                                                                            opt.decay_epoch))

    model = PolypNet(channel=64)

    total = sum([param.nelement() for param in model.get_parameters()])
    print('Number of parameter:%.2fM' % (total / 1e6))

    optimizer = nn.Adam(model.trainable_params(), learning_rate=opt.lr)

    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, opt.train_eg_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=4)
    test_loader = get_loader_test(opt.test_img_dir, opt.test_gt_dir, testsize=opt.trainsize)

    total_step = train_loader.get_dataset_size()
    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                                opt.batchsize, opt.save_model, total_step), '-' * 30)

    train = Trainer(train_loader, test_loader, model, optimizer, opt)
    train.train(opt.epoch)
