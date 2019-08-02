import os
import datetime
import sys
from collections import defaultdict

import numpy as np

import torch
from torch.optim import Adam

import argparse

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


archives = (
    ('team_286/12_idet_food/63', '135', 1.),
    ('team_286/12_idet_food/74', '175', 1.),
    ('team_286/12_idet_food/91', '200', 1.),
)


def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test')

    ensembles_xy = []
    ensembles_w = []
    for sess, chkp, w in archives:
        nsml.load(checkpoint=chkp, session=sess)

        model.eval()
        outputs = []
        outputs_w = []
        num_data = 0
        for idx, (image, _) in enumerate(test_loader):
            with torch.no_grad():
                locs, scores = model(image.cuda())
                all_images_boxes, all_scores = model.detect_objects(locs, scores)

            for box in all_images_boxes:
                box = box.detach().cpu().numpy()
                box_xy = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
                outputs.append(box_xy)
            outputs_w.extend(all_scores)
            num_data += len(all_images_boxes)
        ensembles_xy.append(np.array(outputs))
        ensembles_w.append(outputs_w)

    # ensembles_xy = np.mean(ensembles_xy, axis=0)
    ensemble_result = [None] * len(ensembles_xy[0])
    best_w = defaultdict(lambda: 0)
    for xys, ws in zip(ensembles_xy, ensembles_w):
        for i, (xy, w) in enumerate(zip(xys, ws)):
            if best_w[i] > w:
                continue
            ensemble_result[i] = xy
            best_w[i] = w
    ensembles_xy = np.array(ensemble_result)

    print(ensembles_xy.shape)
    assert ensembles_xy.shape[0] == num_data
    assert ensembles_xy.shape[1] == 4

    ensembles = []
    for xy in ensembles_xy:
        box = np.array([xy[0], xy[1], xy[2] - xy[0], xy[3] - xy[1]])
        ensembles.append(box)

    outputs = np.stack(ensembles, axis=0)
    assert outputs.shape[0] == num_data
    assert outputs.shape[1] == 4
    print(outputs.shape)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.)

    bind_nsml(model, optimizer, scheduler)
    if config.pause:
        nsml.paused(scope=locals())

    nsml.save('final')

    if config.transfer:
        nsml.load(checkpoint='100', session='team_286/12_idet_food/41')
        nsml.save('resave')
        sys.exit(0)
