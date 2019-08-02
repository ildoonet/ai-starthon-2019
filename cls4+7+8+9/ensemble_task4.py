import os
import math
import datetime
import sys
import warnings
from collections import defaultdict

import numpy as np

import time
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.parallel.data_parallel import DataParallel
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

import torchvision.models as models
import argparse

from common import get_logger, accuracy, Accumulator, rand_bbox
from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_with_split, test_ret_loader, CustomDataset
from evaluation import evaluation_metrics

from tqdm import tqdm

from theconf.config import Config as C
from theconf.argument_parser import ConfigArgumentParser

try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')
    VAL_DATASET_PATH = None
except:
    IS_ON_NSML=False
    TRAIN_DATASET_PATH = os.path.join('/media/yoo/Data/NIPAKoreanFoodCls/train/train_data')
    VAL_DATASET_PATH = None

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "^(Palette)", UserWarning)
logger = get_logger('nsml_4_food')


archives = (
    ('team_286/4_cls_food/103', '100', 1.0),
    ('team_286/4_cls_food/162', '100', 0.6),
)


def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(root=os.path.join(root_path, 'test_data'), phase='test')

    res_fcs = []
    for sess, chkp, w in archives:
        nsml.load(checkpoint=chkp, session=sess)

        model.eval()
        res_fc = None
        res_id = None
        for idx, (data_id, image, _) in enumerate(tqdm(test_loader)):
            image = image.cuda()
            with torch.no_grad():
                fc = model(image)
            fc = fc.detach().cpu().numpy()
            fc = np_softmax(fc)

            # with torch.no_grad():
            #     fc2 = model(torch.flip(image, (3, )))       # TTA : horizontal flip
            # fc2 = fc2.detach().cpu().numpy()
            # fc2 = np_softmax(fc2)
            # fc = fc + fc2

            if C.get()['infer_mode'] == 'face':
                fc[:, range(60)] = -1
                # target_lb = list(range(60, 100))

            if idx == 0:
                res_fc = fc
                res_id = data_id
            else:
                res_fc = np.concatenate((res_fc, fc), axis=0)
                res_id = res_id + data_id
        res_fcs.append(res_fc * w)

    res_cls = np.argmax(np.sum(res_fcs, axis=0), axis=1)

    return [res_id, res_cls]


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded', dir_name)

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    # mode argument
    args = ConfigArgumentParser(conflict_handler='resolve')
    args.add_argument("--cv", type=int, default=0)
    args.add_argument("--ratio", type=float, default=0.1)

    # reserved for nsml
    args.add_argument("--cuda", type=bool, default=True)

    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    logger.info(str(C.get().conf))

    num_classes = C.get()['num_class']
    base_lr = config.lr
    cuda = config.cuda
    eval_split = 'val'
    mode = config.mode

    model = models.resnet50(pretrained=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(512 * 4, num_classes)
    loss_fn = nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    if C.get()['opt'] == 'adam':
        optimizer = Adam(model.parameters(), lr=C.get()['lr'], weight_decay=C.get()['optimizer']['decay'])
    elif C.get()['opt'] == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer']['momentum'],
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=(C.get()['optimizer']['decay'] > 0)
        )
    else:
        raise ValueError(C.get()['opt'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epochs'], eta_min=0.)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler)

        if config.pause:
            nsml.paused(scope=locals())

    if mode != 'train':
        sys.exit(0)

    logger.info(archives)
    nsml.save('final')
