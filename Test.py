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

        d = np.zeros((480, 640))
        d[fixations[:,0] ,fixations[:,1]] = 1
        d = d[:,:, None]
        return img, ToTensor()(saliency_map), ToTensor()(d)



from metrics import *
import torch
import torchvision.transforms as transforms

# from dataset import SaliconDataset
from main import ZHANGYiNet_REPRO_1
from models import *

NormMean_imgs = [0.50893384, 0.4930997, 0.46955067]
NormStd_imgs = [0.2652982, 0.26586023, 0.27988392]
img_H = 240
img_W = 320

normTransform = transforms.Normalize(NormMean_imgs, NormStd_imgs)
imgsTransform = transforms.Compose([
    transforms.Resize([img_H, img_W]),
    transforms.ToTensor(), # 0-255 automatically transformed to 0-1
    normTransform
])
salicon_dataset = SaliconDataset("/media/data2/infotech/datasets/salicon/",
                                 "/media/data2/infotech/datasets/COCO/annotations/salicon_instances_val2014.json", transforms=imgsTransform)

PATH = "/home/michalgorszczak/bionn/fixations/lstm_based/samPytorch/Saliency-Attentive-Model-Pytorch/Results/net_params.pkl"

gp = generate_gaussian_prior()
device = torch.device('cuda:1')
model = ZHANGYiNet_REPRO_1(gaussian_prior=gp)
model.to(device)

model.load_state_dict(torch.load(PATH, map_location=device))
# print(salicon_dataset[100])



from torch.utils.data import DataLoader

valid_loader = DataLoader(dataset=salicon_dataset, batch_size=4, shuffle=True, drop_last=True)


criterion_CC = MyCorrCoef()
criterion_CC.to(device)

# build a normalized scanpath saliency
criterion_NSS = MyNormScanSali()
criterion_NSS.to(device)

cc = 0
nss = 0


# auc_shuffled()
auc_s = 0
for val, x in enumerate(valid_loader):
    print(val)
    img, saliency_map, d_map = x
    inputs = img.to(device)
    maps = saliency_map.to(device)
    fixs = d_map.to(device)

    outputs = model(inputs)

    _, _, map_fixations = salicon_dataset[random.randint(0, 4999)]
    fix_points = np.argwhere(map_fixations.cpu().detach().numpy()[0] == 1)

    for _ in range(9):
        img, saliency_map, map_fixations= salicon_dataset[random.randint(0,4999)]
        fix_points = np.vstack((fix_points,  np.argwhere(map_fixations.cpu().detach().numpy()[0] == 1)))
    fix_points = np.argwhere(map_fixations.cpu().detach().numpy()[0] == 1)
    for y in range(outputs.shape[0]):
        true_fixes = np.argwhere(d_map[y].cpu().detach().numpy()[0] == 1)
        auc_s += auc_shuffled(outputs[y].cpu().detach().numpy()[0], true_fixes, fix_points)
    # cc += criterion_CC(outputs, maps).cpu().detach().numpy()
    # nss += criterion_NSS(outputs, fixs).cpu().detach().numpy()


print("auc_s ={}".format(auc_s/ 5000))
# print("nss ={}".format(nss/ len(valid_loader)))

