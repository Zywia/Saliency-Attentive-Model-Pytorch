import pandas as pd
import scipy.io
import torch
import os
import json
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np

class ImageRepository:
    def __init__(self, repository_source):
        with open(repository_source) as jf:
            repository_json = json.load(jf)

        self.__images = pd.DataFrame.from_dict(repository_json['images'])
        self.__annotations = pd.DataFrame.from_dict(repository_json['annotations'])
        self.__categories = pd.DataFrame.from_dict(repository_json['categories'])

    def get_cattegories_present_on_image(self, image_names: list):
        cattegories_present_on_image = []

        for image_name in image_names:
            image_basic_information = self.__images[self.__images.file_name == image_name]
            image_id = image_basic_information.id.iat[0]

            cattegories_present_on_image.append(
                self.__annotations[self.__annotations.image_id == image_id].category_id.to_numpy())
        return cattegories_present_on_image

    def get_all_categories(self):
        return self.__categories.id.to_numpy()

    def categories_ids_to_name(self, categories_ids):
        return self.__categories[['id', 'name']].set_index('id').loc[categories_ids]


class SaliconDataset(Dataset):
    def __init__(self, root_dir, coco_val_json_file_path, transforms=None):
        self.__transforms = transforms
        self.__encoder = MultiLabelBinarizer()
        self.__saliency_maps_format = ".png"
        self.__fixations_format = ".mat"
        self.__root_dir = root_dir
        # self.__image_sources = ["train", "val"]
        self.__images_path = "images"
        self.__saliency_maps_path = "maps"
        self.__fixations_path = "fixations"

        self.__image_repository =  ImageRepository(coco_val_json_file_path)

        list_of_images = os.listdir(os.path.join(root_dir, self.__images_path, "val"))
        self.__images_with_parent_path = list(map(lambda x: os.path.join("val", x), list_of_images))

    def get_fixation(raw_fixation: np.ndarray, by_subject: bool = False) -> np.ndarray:
        """
        Process raw salicon human fixation data onto a array with fixation points from all subjects
        :param raw_fixation: np.array, raw data from salicon
        :param by_subject: TODO: write me
        :return: np.ndarray, human fixation points in array of shape (N, 2)
        """
        fixation_points = []

        # go through the data and save the fixation points
        for gazes, time_stamps, fix_points in raw_fixation['gaze'].flatten():
            # our image is height x width and coords are x,y
            fix_points[:, [0, 1]] = fix_points[:, [1, 0]]
            # fix matlab indexing
            fix_points -= 1
            fixation_points.append(fix_points.astype(np.int64))

        if not by_subject:
            # join the points from all subjects
            fixation_points = np.concatenate(fixation_points, axis=0)
            # remove duplicates
            fixation_points = np.unique(fixation_points, axis=0)

        return fixation_points

    def __len__(self):
        return len(self.__images_with_parent_path)

    def decode(self, encoded_labels):
        return self.__encoder.inverse_transform(encoded_labels)

    def __getitem__(self, idx):
        image_with_parent_folder = self.__images_with_parent_path[idx]
        saliency_map = image_with_parent_folder.split(".")[0] + self.__saliency_maps_format
        fixation_with_parent_folder = image_with_parent_folder.split(".")[0] + self.__fixations_format
        img = Image.open(os.path.join(self.__root_dir, self.__images_path, image_with_parent_folder)).convert('RGB')
        saliency_map = Image.open(os.path.join(self.__root_dir, self.__saliency_maps_path, saliency_map))
        fixations = scipy.io.loadmat(os.path.join(self.__root_dir, self.__fixations_path, fixation_with_parent_folder))
        fixations = SaliconDataset.get_fixation(fixations, by_subject=False)

        if self.__transforms is not None:
            img = self.__transforms(img)

        return img, ToTensor()(saliency_map), fixations