import random

import numpy as np
from torch.utils import data
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import torch
import os

from utils import flip, photometric_distort


def get_transform():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.Resize((300, 300)))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)


def target_transform(target, sizes):
    width, height = sizes

    target[0] = float(target[0]) / float(width)
    target[1] = float(target[1]) / float(height)
    # target[2] = target[0] + float(target[2]) / float(width)
    # target[3] = target[1] + float(target[3]) / float(height)
    target[2] = max(0.0001, float(target[2]) / float(width))
    target[3] = max(0.0001, float(target[3]) / float(height))

    return target.unsqueeze(0)


class CustomDataset(data.Dataset):
    def __init__(self, root, transform, target_transform, loader=default_loader, aug=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.aug = aug

        self.images = []
        self.boxes = []
        dir_list = sorted(os.listdir(self.root))
        self.ext = '.box' if '_car' in root else '.txt'

        for file_path in dir_list:
            if file_path.endswith('.jpg'):
                self.images.append(os.path.join(self.root, file_path))

                box_file_path = file_path[:-4] + self.ext
                box_str = open(os.path.join(self.root, box_file_path), 'r', encoding='utf-8').read().split('\n')[0]
                box = [int(float(bb)) for bb in box_str.split(',')]
                self.boxes.append(box)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = self.loader(self.images[index])
        targets = torch.Tensor(self.boxes[index])

        if self.aug:
            images = photometric_distort(images)
            if random.random() < 0.5:
                images, targets = flip(images, targets)

        inputs = self.transform(images)
        width, height = images.size
        targets = self.target_transform(targets, (width, height))

        return inputs, targets


def data_loader(root, phase='train', batch_size=256):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError
    input_transform = get_transform()
    dataset = CustomDataset(root, input_transform, target_transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=4, pin_memory=True,
                           shuffle=is_train)


def data_loader_with_split(root, train_split=0.9, batch_size=256, val_label_file='./val_label'):
    input_transform = get_transform()
    dataset_tr = CustomDataset(root, input_transform, target_transform, aug=True)
    dataset_vl = CustomDataset(root, input_transform, target_transform, aug=False)
    split_size = int(len(dataset_tr) * train_split)

    random.seed(1958)
    l = list(range(len(dataset_tr)))
    random.shuffle(l)
    train_idxs = l[:split_size]
    valid_idxs = l[split_size:]
    train_set = Subset(dataset_tr, train_idxs)
    valid_set = Subset(dataset_vl, valid_idxs)
    # train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])
    print(len(train_set), len(valid_set))
    tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                num_workers=4, pin_memory=True,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 num_workers=4, pin_memory=True,
                                 shuffle=False)

    gt_labels = [valid_set[idx][1] for idx in range(len(valid_set))]
    gt_labels_string = [','.join([str(s.numpy()) for s in l]) for l in list(gt_labels)]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))

    print('data_loader_with_split-')
    return tr_loader, val_loader, val_label_file
