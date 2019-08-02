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


def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(root=os.path.join(root_path, 'test_data'), phase='test')

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

    res_cls = np.argmax(res_fc, axis=1)

    return [res_id, res_cls]


def _infer_ret(model, root_path, test_loader=None, local_val=False):
    """
    모델과 데이터가 주어졌을 때, 다음과 같은 데이터 구조를 반환하는 함수를 만들어야 합니다.

    [ [ query_image_id_1, predicted_database_image_id_1 ],
      [ query_image_id_2, predicted_database_image_id_2 ],
      ...
      [ query_image_id_N, predicted_database_image_id_N ] ]

    README 설명에서처럼 predicted_database_image_id_n 은 query_image_id_n 에 대해
    평가셋 이미지 1,...,n-1,n+1,...,N 를 데이터베이스로 간주했을 때에 가장 쿼리와 같은 카테고리를
    가질 것으로 예측하는 이미지입니다. 이미지 아이디는 test_loader 에서 extract 되는 첫번째
    인자인 data_id 를 사용합니다.

    Args:
      model: 이미지를 인풋으로 받아서 feature vector를 반환하는 모델
      root_path: 데이터가 저장된 위치
      test_loader: 사용되지 않음
      local_val: 사용되지 않음

    Returns:
      top1_reference_ids: 위에서 설명한 list 데이터 구조
    """
    def new_forward(self, x, extract=False):
        if isinstance(self, DataParallel):
            self = self.module
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat2 = torch.flatten(x, 1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x
        x = self.fc(x)
        x = softmax(x, dim=1)

        if C.get()['feat'] == 'feat2':
            return feat2
        elif C.get()['feat'] == 'pool':
            return feat
        else:
            return x

    if test_loader is None:
        test_loader = test_ret_loader(root=os.path.join(root_path, 'test_data'))

    # TODO 모델의 아웃풋을 적당히 가공하고 연산하여 각 query에 대해 매치가 되는 데이터베이스
    # TODO 이미지의 ID를 찾는 모듈을 구현 (현재 구현은 베이스라인 - L2 정규화 및 내적으로 가장
    # TODO 비슷한 이미지 조회).
    model.eval()
    feats = None
    data_ids = None
    s_t = time.time()
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        with torch.no_grad():
            feat = new_forward(model, image, extract=True)
        feat = feat.detach().cpu().numpy()
        feat = feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]

        # with torch.no_grad():
        #     feat2 = new_forward(model, torch.flip(image, (3, )), extract=True)
        # feat2 = feat2.detach().cpu().numpy()
        # feat2 = feat2 / np.linalg.norm(feat2, axis=1)[:, np.newaxis]
        # feat = feat + feat2

        if feats is None:
            feats = feat
            data_ids = data_id
        else:
            feats = np.append(feats, feat, axis=0)
            data_ids = np.append(data_ids, data_id, axis=0)

        if time.time() - s_t > 10:
            print('Infer batch {}/{}.'.format(idx + 1, len(test_loader)))

    score_matrix = feats.dot(feats.T)
    np.fill_diagonal(score_matrix, -np.inf)
    top1_reference_indices = np.argmax(score_matrix, axis=1)
    top1_reference_ids = [
        [data_ids[idx], data_ids[top1_reference_indices[idx]]] for idx in
        range(len(data_ids))
    ]

    return top1_reference_ids


def local_eval(model, test_loader=None, test_label_file=None):
    model.eval()
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file)
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

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path, top_k=1):
        if C.get()['mode'] != 'test':
            return _infer(model, root_path)
        if C.get()['infer_mode'] == 'ret':
            return _infer_ret(model, root_path)
        else:
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
    args.add_argument("--transfer", type=bool, default=False)

    config = args.parse_args()

    logger.info(str(C.get().conf))

    num_classes = C.get()['num_class']
    base_lr = config.lr
    cuda = config.cuda
    eval_split = 'val'
    mode = config.mode

    if C.get()['model'] == 'resnet18':
        model = models.resnet18(pretrained=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(512 * 1, num_classes)
    elif C.get()['model'] == 'resnet50':
        model = models.resnet50(pretrained=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(512 * 4, num_classes)
    elif C.get()['model'] == 'resnet101':
        model = models.resnet101(pretrained=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(512 * 4, num_classes)
    else:
        raise ValueError(C.get()['model'])
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

    # if C.get()['infer_mode'] == 'face':
    #     targets_only = []
    #     lbs = CustomDataset(TRAIN_DATASET_PATH).targets
    #     for lb_id in range(num_classes):
    #         if lbs.count(lb_id) > 150:
    #             continue
    #         targets_only.append(lb_id)
    #     print(targets_only)

    if config.transfer:
        # nsml.load(checkpoint='transfer', session='team_286/4_cls_food/89')
        nsml.load(checkpoint='100', session='team_286/4_cls_food/103')  # cv=1 cutmix 0.5
        # nsml.load(checkpoint='55', session='team_286/7_icls_face/2')
        # nsml.load(checkpoint='transfer', session='team_286/8_iret_food/12')
        # nsml.load(checkpoint='20', session='team_286/9_iret_car/16')
        nsml.save('resave')
        sys.exit(0)

    tr_loader, val_loader, val_label = data_loader_with_split(root=TRAIN_DATASET_PATH, cv_ratio=config.ratio, cv=config.cv, batch_size=C.get()['batch'])
    time_ = datetime.datetime.now()
    best_val_top1 = 0

    dataiter = iter(tr_loader)
    num_steps = 100000 // C.get()['batch']

    from pystopwatch2 import PyStopwatch

    for epoch in range(C.get()['epochs']):
        w = PyStopwatch()
        metrics = Accumulator()
        scheduler.step()
        model.train()
        cnt = 0
        for iter_ in range(num_steps):
            w.start(tag='step1')
            _, x, label = next(dataiter)
            if cuda:
                x, label = x.cuda(), label.cuda()

            w.pause(tag='step1')
            cutmix = C.get().conf.get('cutmix', defaultdict(lambda: 0.))
            cutmix_alpha = cutmix['alpha']
            cutmix_prob = cutmix['prob']

            if cutmix_alpha <= 0.0 or np.random.rand(1) > cutmix_prob:
                pred = model(x)
                loss = loss_fn(pred, label)
            else:
                # CutMix : generate mixed sample
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                rand_index = torch.randperm(x.size()[0]).cuda()
                target_a = label
                target_b = label[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

                pred = model(x)
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                loss = loss_fn(pred, target_a) * lam + loss_fn(pred, target_b) * (1. - lam)
            w.start(tag='step3')

            top1, top5 = accuracy(pred, label, (1, 5))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()
            w.pause(tag='step3')

            metrics.add_dict({
                'loss': loss.item() * len(x),
                'top1': top1.item() * len(x),
                'top5': top5.item() * len(x),
            })
            cnt += len(x)
        model.eval()

        postfix = metrics / cnt
        logger.info('[{:d}/{:d}] lr({:.6f}) loss({:.4f}) top1({:.3f}) top5({:.3f})'.format(
            epoch + 1, C.get()['epochs'], optimizer.param_groups[0]['lr'], postfix['loss'], postfix['top1'], postfix['top5'])
        )
        logger.info(str(w))

        if (epoch + 1) % 5 == 0:
            val_top1 = local_eval(model, val_loader, val_label)
            if IS_ON_NSML:
                nsml.save(str(epoch + 1))
                nsml.report(summary=True, scope=locals(), step=(epoch + 1), tr_top1=postfix['top1'], val_top1=val_top1)
            logger.info('validation epoch=%d top1=%4f' % (epoch + 1, val_top1))

            if best_val_top1 < val_top1 and IS_ON_NSML:
                best_val_top1 = val_top1
                logger.info('best @ %d' % (epoch + 1))
                nsml.save('best')
