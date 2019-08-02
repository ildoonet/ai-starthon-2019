import os
import datetime
import sys

import numpy as np

import torch
from torch.optim import Adam

import argparse

from torch.optim.sgd import SGD

from common import Accumulator, get_logger
from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_with_split
from evaluation import evaluation_metrics

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from model import SSD300, MultiBoxLoss

if IS_ON_NSML:
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')
    VAL_DATASET_PATH = None
else:
    TRAIN_DATASET_PATH = os.path.join('/home/data/NIPAKoreanFoodLocalizeSmall/train/train_data')
    VAL_DATASET_PATH = os.path.join('/home/data/NIPAKoreanFoodLocalizeSmall/test')


logger = get_logger('det')


def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test')

    model.eval()
    outputs = []
    for idx, (image, _) in enumerate(test_loader):
        with torch.no_grad():
            locs, scores = model(image.cuda())
            all_images_boxes, _ = model.detect_objects(locs, scores)

        for box in all_images_boxes:
            box = box.detach().cpu().numpy()
            outputs.append(box)

    outputs = np.stack(outputs, axis=0)
    return outputs


def local_eval(model, test_loader=None, test_label_file=None):
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file
    )
    logger.info('Eval result: {:.4f} mIoU'.format(metric_result))
    return metric_result


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        logger.info('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        logger.info('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--batch", type=int, default=64)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=200)
    args.add_argument("--eval_split", type=str, default='val')
    args.add_argument("--transfer", type=int, default=0)

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    train_split = config.train_split
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    eval_split = config.eval_split
    mode = config.mode

    model = SSD300(n_classes=2)
    loss_fn = MultiBoxLoss(priors_cxcy=model.priors_cxcy)

    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=base_lr, weight_decay=1e-4)
    # optimizer = SGD([param for param in model.parameters() if param.requires_grad], lr=base_lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.)

    bind_nsml(model, optimizer, scheduler)
    if config.pause:
        nsml.paused(scope=locals())

    if config.transfer:
        nsml.load(checkpoint='100', session='team_286/12_idet_food/41')
        nsml.save('resave')
        sys.exit(0)

    if mode == 'train':
        tr_loader, val_loader, val_label_file = data_loader_with_split(root=TRAIN_DATASET_PATH, train_split=train_split, batch_size=config.batch)
        time_ = datetime.datetime.now()
        num_batches = len(tr_loader)

        local_eval(model, val_loader, val_label_file)
        best_iou = 0.
        for epoch in range(num_epochs):
            metrics = Accumulator()
            scheduler.step()
            model.train()
            cnt = 0
            for iter_, data in enumerate(tr_loader):
                x, label = data
                label[:, :, 2:] = label[:, :, 2:] + label[:, :, :2]  # convert to min-xy, max-xy

                if cuda:
                    x = x.cuda()
                    label = label.cuda()

                predicted_locs, predicted_scores = model(x)
                loss = loss_fn(predicted_locs, predicted_scores, label, torch.ones((label.shape[0], 1), dtype=torch.long))
                if torch.isnan(loss):
                    logger.error('loss nan. epoch=%d iter=%d' % (epoch, iter_))
                    sys.exit(-1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics.add_dict({
                    'loss': loss.item() * len(x)
                })
                cnt += len(x)

                if iter_ >= 50:  # TODO
                    break

            postfix = metrics / cnt
            logger.info('[{:d}/{:d}] lr({:.6f}) loss({:.4f})'.format(
                epoch + 1, num_epochs, optimizer.param_groups[0]['lr'], postfix['loss'])
            )

            if (epoch + 1) % 5 == 0:
                miou = local_eval(model, val_loader, val_label_file)
                if best_iou < miou:
                    best_iou = miou
                    nsml.save('best')
                    nsml.report(summary=True, scope=locals(), step=(epoch + 1), loss=postfix['loss'], miou=miou)
                nsml.save(str(epoch + 1))
                nsml.report(summary=True, scope=locals(), step=(epoch + 1), loss=postfix['loss'], miou=miou)

            elapsed = datetime.datetime.now() - time_
