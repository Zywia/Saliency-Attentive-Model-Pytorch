from __future__ import division
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
import torch
from scipy.io import loadmat
from config import shape_r_out, shape_c_out
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# complement the class of Mydataset
class MyDataset(Dataset):
    def __init__(self, imgs_txt_path, maps_txt_path,  fixs_txt_path,
                 transform_img = None, transform_map = None, transform_fix = None):
        fh_imgs = open(imgs_txt_path, 'r')
        fh_maps = open(maps_txt_path, 'r')
        fh_fixs = open(fixs_txt_path, 'r')
        imgs = []
        maps = []
        fixs = []
        for line in fh_imgs:
            line = line.rstrip()
            imgs.append(line)
        for line in fh_maps:
            line = line.rstrip()
            maps.append(line)
        for line in fh_fixs:
            line = line.rstrip()
            fixs.append(line)

        self.imgs = imgs
        self.maps = maps
        self.fixs = fixs
        self.transform_img = transform_img
        self.transform_map = transform_map
        self.transform_fix = transform_fix

    def __get_fixation(raw_fixation: np.ndarray) -> np.ndarray:
        fixation_points = []

        # go through the data and save the fixation points
        for gazes, time_stamps, fix_points in raw_fixation['gaze'].flatten():
            # our image is height x width and coords are x,y

            fix_points[:, [0, 1]] = fix_points[:, [1, 0]]
            # fix matlab indexing
            fix_points -= 1
            fixation_points.append(fix_points.astype(np.int64))


        # join the points from all subjects
        fixation_points = np.concatenate(fixation_points, axis=0)
        # remove duplicates
        fixation_points = np.unique(fixation_points, axis=0)

        return fixation_points

    def __getitem__(self, index):
        img_path_cur = self.imgs[index]
        map_path_cur = self.maps[index]
        fix_path_cur = self.fixs[index]
        img = Image.open(img_path_cur).convert('RGB')
        map = Image.open(map_path_cur)
        fix = loadmat(fix_path_cur)
        fix = MyDataset.__get_fixation(fix)
        fix_map = np.zeros((shape_r_out, shape_c_out), dtype=np.uint8)
        fix_map[fix[:, 0], fix[:, 1]] = 255
        fix = transforms.ToPILImage()(fix_map)

        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_map is not None:
            map = self.transform_map(map)
        if self.transform_fix is not None:
            fix = self.transform_fix(fix)

        return img, map, fix

    def __len__(self):
        return len(self.imgs)


# add temporal dimention
def format_attLSTM(x, nb_ts):

    y = []
    x.unsqueeze_(1)
    for i in range(nb_ts):
        y.append(x)
    feature = torch.cat(y,1)

    return feature


# preprocessing the fixs dataset
def fixs_preprocessing(fixs):

    fixs_out = []
    for i in range(fixs.shape[0]):
       img = transforms.ToPILImage()(fixs[i])
       plt.imshow(img)
       plt.show()

    print('Done!')