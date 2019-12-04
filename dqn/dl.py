import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .pietorch import data_convertors


def get_dataloader(ds, data_dir, noise=10, crop=256, jpeg_quality=40):
    transform = T.ToTensor()

    if ds == 'RealNoiseHKPoly':
        crop_size, with_data_aug  = 128, False
        test_root, test_list_pth = os.path.join(data_dir, ds, 'test1/'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, 'test1_list.txt')
        test_convertor = data_convertors.ConvertImageSet(test_root, test_list_pth, ds,
                                                transform=transform,crop_size=crop_size)

        data_root, imlist_pth  = os.path.join(data_dir, ds, 'OriginalImages/'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, 'train_list.txt')
        convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, ds,
                                            transform=transform, is_train=True,
                                            with_aug=with_data_aug, crop_size=crop_size)

        dataloader, test_dataloader = DataLoader(convertor, batch_size=32, shuffle=False), DataLoader(test_convertor, batch_size=32, shuffle=False)
    elif ds == 'GoPro':
        transform = [jpeg_quality, T.ToTensor(), noise]
        crop_size, with_data_aug  = crop, False
        test_root, test_list_pth = os.path.join(data_dir, ds, 'test/'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, 'test_list.txt')
        test_convertor = data_convertors.ConvertImageSet(test_root, test_list_pth, ds,
                                                transform=transform, resize_to=(640, 360), crop_size=crop_size)

        data_root, imlist_pth  = os.path.join(data_dir, ds, 'train/'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, 'train_list.txt')
        

        convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, ds,
                                            transform=transform, is_train=True,
                                            with_aug=with_data_aug, resize_to=(640, 360), crop_size=crop_size)
        dataloader, test_dataloader = DataLoader(convertor, batch_size=1, shuffle=False), DataLoader(test_convertor, batch_size=32, shuffle=False)
    
    elif ds == 'RainDrop':
        test_set = 'test_a'
        crop_size, with_data_aug  = crop, False
        transform = [jpeg_quality, T.ToTensor(), noise]

        test_root, test_list_pth = os.path.join(data_dir, ds, test_set, test_set), os.path.join('/home/shumin/projects/ma-dqn','split', ds, test_set+'_list.txt')
        test_convertor = data_convertors.ConvertImageSet(test_root, test_list_pth, ds, transform=transform, crop_size=crop_size)

        data_root, imlist_pth  = os.path.join(data_dir, ds, 'train', 'train/'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, 'train_list.txt')
        
        convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, ds,
                                            transform=transform, is_train=True,
                                            with_aug=with_data_aug, crop_size=crop_size)
        dataloader, test_dataloader = DataLoader(convertor, batch_size=32, shuffle=False), DataLoader(test_convertor, batch_size=32, shuffle=False)
    
    elif ds == 'RESIDE':
        test_root, test_list_pth = os.path.join(data_dir, ds, 'test_a', 'test_a'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, test_a+'_list.txt')
        test_convertor = data_convertors.ConvertImageSet(test_root, test_list_pth, ds, transform=transform)

        data_root, imlist_pth  = os.path.join(data_dir, ds, 'train', 'train/'), os.path.join('/home/shumin/projects/ma-dqn','split', ds, 'train_list.txt')
        crop_size, with_data_aug  = 256, False

        convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, ds,
                                            transform=transform, is_train=True,
                                            with_aug=with_data_aug, crop_size=crop_size)
        dataloader, test_dataloader = DataLoader(convertor, batch_size=32, shuffle=False), DataLoader(test_convertor, batch_size=32, shuffle=False)


    return dataloader, test_dataloader