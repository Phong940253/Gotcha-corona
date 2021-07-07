import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

DIR_INPUT = '/kaggle/input/gotcha-corona'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        records = records.reset_index(drop=True)
        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is six one class
        labels = torch.as_tensor(records['labels'].values, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(
                tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

# Albumentations


def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


class FasterRCNN:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False)
        num_classes = 6  # 1 class (wheat) + background
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(
            "./Corona/fasterrcnn_resnet50_fpn.pth"))
        self.model.eval()
        self.transforms = get_valid_transform()

    def eval(self, img, visuable=False):
        sample = {
            'image': img,
            'bboxes': [],
            'labels': []
        }
        sample = self.transforms(**sample)
        image = sample['image'].permute(1, 2, 0).cpu().numpy()

        target = {}
        target['boxes'] = []
        target['labels'] = []
        target['image_id'] = []
        target['area'] = []
        target['iscrowd'] = []

        images = list(image.to(self.device) for image in [sample['image']])

        cpu_device = torch.device("cpu")
        outputs = self.model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]

        boxes = outputs[0]['boxes'].detach().numpy().astype(np.int32)
        labels = outputs[0]['labels'].detach().numpy().astype(np.int32)
        scores = outputs[0]['scores'].detach().numpy().astype(np.float32)
        print(scores)
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        if visuable:
            mapLabel = {
                "type1": 0,
                "type2": 1,
                "type3": 2,
                "queen": 3,
                "patient": 4,
                "doctor": 5
            }
            for ind, val in enumerate(boxes):
                if labels[ind] == 4 or labels[ind] == 5:
                    color = (220, 0, 0)
                else:
                    color = (0, 220, 0)

                type = list(mapLabel.keys())[
                    list(mapLabel.values()).index(labels[ind])]
                cv2.rectangle(image,
                              (val[0], val[1]),
                              (val[2], val[3]),
                              color, 3)
                cv2.putText(image, type, (val[0], val[1] - 5), 0, 0.3, color)
                cv2.putText(image, str(
                    scores[ind]), (val[2] - 20, val[3] + 10), 0, 0.3, color)
            ax.set_axis_off()
            ax.imshow(image)
            plt.show()
