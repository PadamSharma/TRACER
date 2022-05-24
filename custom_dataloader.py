import cv2
import glob
import torch
import numpy as np
import albumentations as albu
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
from PIL import Image


class DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, edge_folder, phase: str = 'train', transform=None, seed=None):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.edges = sorted(glob.glob(edge_folder + '/*'))
        self.transform = transform

        train_images, val_images, train_gts, val_gts, train_edges, val_edges = train_test_split(self.images, self.gts,
                                                                                                self.edges,
                                                                                                test_size=0.2,
                                                                                                random_state=seed)
        if phase == 'train':
            self.images = train_images
            self.gts = train_gts
            self.edges = train_edges
        elif phase == 'val':
            self.images = val_images
            self.gts = val_gts
            self.edges = val_edges
        else:  # Testset
            pass

    def transform_random_crop(self, image, mask, edge):
        # print(image.size)
        if random.random() > 0.3:
            resize = transforms.Resize(size=(640, 640))
            image = resize(image)
            mask = resize(mask)
            edge = resize(edge)
        else:
            resize = transforms.Resize(size=(960, 960))
            image = resize(image)
            mask = resize(mask)
            edge = resize(edge)

            i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(640, 640))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            edge = TF.crop(edge, i, j, h, w)
        # print(image.size)
    
        return image, mask, edge

    def __getitem__(self, idx):
        # image = cv2.imread(self.images[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.gts[idx])
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # edge = cv2.imread(self.edges[idx])
        # edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
  
        image = Image.open(self.images[idx])
        mask = Image.open(self.gts[idx])
        edge = Image.open(self.edges[idx])
        image, mask, edge = self.transform_random_crop(image, mask, edge)

        image = np.array(image)
        mask = np.array(mask)
        edge = np.array(edge)

        # print(image.shape)

        cv2.imwrite('/content/TRACER/data/Test/images/'+ (self.images[idx]).split('/')[-1], image)
        cv2.imwrite('/content/TRACER/data/Test/masks/'+  (self.gts[idx]).split('/')[-1], mask)
        cv2.imwrite('/content/TRACER/data/Test/edges/'+  (self.edges[idx]).split('/')[-1], edge)



        if self.transform is not None:
            augmented = self.transform(image=image, masks=[mask, edge])
            image = augmented['image']
            mask = np.expand_dims(augmented['masks'][0], axis=0)  # (1, H, W)
            mask = mask / 255.0
            edge = np.expand_dims(augmented['masks'][1], axis=0)  # (1, H, W)
            edge = edge / 255.0

        return image, mask, edge

    def __len__(self):
        return len(self.images)


class Test_DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, transform=None):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.transform = transform

    def __getitem__(self, idx):
        image_name = Path(self.images[idx]).stem
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        gt_image = None if self.gts is None else self.gts[idx]
        return image, gt_image, original_size, image_name

    def __len__(self):
        return len(self.images)


def get_loader(img_folder, gt_folder: str, edge_folder, phase: str, batch_size, shuffle,
               num_workers, transform, seed=None):
    if phase == 'test':
        dataset = Test_DatasetGenerate(img_folder, gt_folder, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        dataset = DatasetGenerate(img_folder, gt_folder, edge_folder, phase, transform, seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 drop_last=True)

    print(f'{phase} length : {len(dataset)}')

    return data_loader


def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if ver == 2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transforms


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt
