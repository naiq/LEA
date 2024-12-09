import os, glob, random, cv2, json, pickle, copy, requests, io, csv
import torch
import numpy as np
from einops import rearrange
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
from matplotlib import pyplot as plt
from functools import partial
from PIL import ImageFilter
from collections import Counter
import open_clip
from torchvision.models import ViT_H_14_Weights, vit_h_14
from collections import Counter
from itertools import combinations



def identity(x):
    return x

def fmri_transform(x, drop_rate=0.2):
    # x: [1, num_voxels]

    x_aug = copy.deepcopy(x)

    idx = np.random.choice(x.shape[1], int(x.shape[1]*drop_rate), replace=False)
    x_aug[0, idx] = 0

    return torch.FloatTensor(x_aug)

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label # img_label: [(label_in_imagenet, category, label_in_training_set), ()]

def reorganize_train_test(train_img, train_fmri, test_img, test_fmri, train_lb, test_lb,
                    test_category, train_index_lookup):
    test_img_ = []
    test_fmri_ = []
    test_lb_ = []
    train_idx_list = []
    num_per_category = 8
    for c in test_category:
        c_idx = c * num_per_category + np.random.choice(num_per_category, 1)[0]
        train_idx = train_index_lookup[c_idx]
        test_img_.append(train_img[train_idx])
        test_fmri_.append(train_fmri[train_idx])
        test_lb_.append(train_lb[train_idx])
        train_idx_list.append(train_idx)

    train_img_ = np.stack([img for i, img in enumerate(train_img) if i not in train_idx_list])
    train_fmri_ = np.stack([fmri for i, fmri in enumerate(train_fmri) if i not in train_idx_list])
    train_lb_ = [lb for i, lb in enumerate(train_lb) if i not in train_idx_list] + test_lb

    train_img_ = np.concatenate([train_img_, test_img], axis=0)
    train_fmri_ = np.concatenate([train_fmri_, test_fmri], axis=0)

    test_img_ = np.stack(test_img_)
    test_fmri_ = np.stack(test_fmri_)
    return train_img_, train_fmri_, test_img_, test_fmri_, train_lb_, test_lb_


def create_Kamitani_CLIP_dataset(path='data/Kamitani/npz',  roi='VC', patch_size=16, image_size=256, image_norm=False, random_flip=False,
            drop_rate=0, subjects = ['sbj_1'], test_category=None, include_nonavg_test=False,
            clip_name='', clip_ckpt='', clip_cache=''):
    img_npz = dict(np.load(os.path.join(path, 'images_256.npz'))) 
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)

    with open(os.path.join(path, 'imagenet_training_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_training_filename = [row for row in csvreader]

    with open(os.path.join(path, 'imagenet_testing_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_testing_filename = [row for row in csvreader]

    train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
    test_img_label, _ = get_img_label(img_class_index, img_testing_filename, naive_label_set) 

    test_img = [] 
    train_img = [] 
    train_fmri = []
    test_fmri = []
    train_img_label_all = []
    test_img_label_all = []
    for sub in subjects:
        npz = dict(np.load(os.path.join(path, f'{sub}.npz')))
        test_img.append(img_npz['test_images'])
        train_img.append(img_npz['train_images'][npz['arr_3']]) 
        train_lb = [train_img_label[i] for i in npz['arr_3']] 
        test_lb = test_img_label

        # roi_mask = npz[roi]
        # tr = npz['arr_0'][..., roi_mask] # train, [1200, 4643]
        # tt = npz['arr_2'][..., roi_mask] # test, [50, 4643]
        # if include_nonavg_test:
        #     tt = np.concatenate([tt, npz['arr_1'][..., roi_mask]], axis=0)

        assert roi == 'VC'
        roi_mask_re = {}
        tr_list = []
        tt_list = []
        tt_list_nonavg = []
        voxel_patches = []
        temp = np.zeros_like(npz['VC'])
        for k in list(npz.keys())[:10]:
            new_mask = npz[k] ^ (temp * npz[k])
            if new_mask.sum() > 0:
                roi_mask_re[k] = new_mask
                tr_list.append(npz['arr_0'][..., new_mask])
                tt_list.append(npz['arr_2'][..., new_mask])
                voxel_patches.append(new_mask.sum())
                if include_nonavg_test:
                    tt_list_nonavg.append(npz['arr_1'][..., new_mask])
            temp += npz[k]
            np.clip(temp, a_min=0, a_max=1)
        tr = np.concatenate(tr_list, axis=1)
        tt = np.concatenate(tt_list, axis=1)
        if include_nonavg_test:
            tt_nonavg = np.concatenate(tt_list_nonavg, axis=1)
            tt = np.concatenate([tt, tt_nonavg], axis=0)
     

        tr = normalize(tr) 
        tt = normalize(tt, np.mean(tr), np.std(tr)) 

        train_fmri.append(tr)
        test_fmri.append(tt)
        if test_category is not None:
            train_img_, train_fmri_, test_img_, test_fmri_, train_lb, test_lb = reorganize_train_test(train_img[-1], train_fmri[-1],
                                                            test_img[-1], test_fmri[-1], train_lb, test_lb,
                                                            test_category, npz['arr_3'])
            train_img[-1] = train_img_
            train_fmri[-1] = train_fmri_
            test_img[-1] = test_img_
            test_fmri[-1] = test_fmri_

        train_img_label_all += train_lb
        test_img_label_all += test_lb

    len_max = max([i.shape[-1] for i in test_fmri])
    test_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in test_fmri]
    train_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in train_fmri]

    test_fmri = np.concatenate(test_fmri, axis=0)
    train_fmri = np.concatenate(train_fmri, axis=0)
    test_img = np.concatenate(test_img, axis=0)
    train_img = np.concatenate(train_img, axis=0)
    num_voxels = train_fmri.shape[-1]

    image_transform_list = [transforms.Resize((image_size, image_size))]
    image_transform_list.append(transforms.ToTensor())
    if image_norm:
        image_transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    image_transform = transforms.Compose(image_transform_list)

    return (Kamitani_CLIP_dataset(train_fmri, train_img, train_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_0']), clip_name, clip_ckpt, clip_cache, phase='test'),
            Kamitani_CLIP_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_2']), clip_name, clip_ckpt, clip_cache, phase='test'))


class Kamitani_CLIP_dataset(torch.utils.data.Dataset):
    def __init__(self, fmri, image, img_label, fmri_transform=identity, image_transform=identity, num_voxels=0, num_per_sub=50,
                 clip_name='', clip_ckpt='', clip_cache='', phase='train'):
        super(Kamitani_CLIP_dataset, self).__init__()
        self.fmri = fmri
        self.image = image 
        if len(self.image) != len(self.fmri):
            self.image = np.repeat(self.image, 35, axis=0)
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.num_per_sub = num_per_sub
        self.img_class = [i[0] for i in img_label]
        self.img_class_name = [i[1] for i in img_label]
        self.naive_label = [i[2] for i in img_label]
        self.return_image_class_info = True

        _, train_preprocess, test_preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=clip_ckpt)
        self.image_preprocess = train_preprocess if phase == 'train' else test_preprocess

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        fmri = self.fmri[index]
        if index >= len(self.image):
            img = np.zeros_like(self.image[0])
        else:
            img = Image.fromarray(self.image[index])

        fmri = np.expand_dims(fmri, axis=0) # (1, num_voxels)

        if self.return_image_class_info:
            img_class = self.img_class[index]
            img_class_name = self.img_class_name[index]
            naive_label = torch.tensor(self.naive_label[index])
            return {'fmri': self.fmri_transform(fmri), 'fmri_target': torch.FloatTensor(fmri), 
                    'image': self.image_transform(img), 'image4clip': self.image_preprocess(img),
                    'image_class': img_class, 'image_class_name': img_class_name, 'naive_label':naive_label}
        else:
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img), 'image4clip': self.image_preprocess(img)}

    def create_iterator(self, batch_size):
        sample_loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            shuffle=True
        )

        for item in sample_loader:
            yield item

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                if k in ['fmri', 'image', 'label', 'image_class', 'image_class_name', 'naive_label']:
                    res[k] = default_collate(temp_)
                else:
                    res[k] = temp_
            else:
                res[k] = None
        return res


def create_BOLD5000_CLIP_dataset(path='../../../data/BOLD5000', patch_size=16, image_size=256, image_norm=False, random_flip=False,
            drop_rate=0, subjects=['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False, clip_name='', clip_ckpt='', clip_cache=''):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()

    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []

    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        fmri_data_sub = normalize(fmri_data_sub) # [5254, 1696]

        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files] # 5254, [([256, 256, 3], cate), ([256, 256, 3], cate), ...]
    
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx]) # [113, 1696]
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx]) # [113, 256, 256, 3]

        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test: # [avg_test; nonavg_test]
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)

        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx]) # [4803, 256, 256, 3]
        train_fmri = fmri_data_sub[train_idx] # [4803, 1696]
        
        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)

    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    image_transform_list = [transforms.Resize((image_size, image_size))]
    image_transform_list.append(transforms.ToTensor())
    if image_norm:
        image_transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    image_transform = transforms.Compose(image_transform_list)

    num_voxels = fmri_train_major.shape[-1]
    return (BOLD5000_CLIP_dataset(fmri_train_major, img_train_major, torch.FloatTensor, image_transform, num_voxels, clip_name, clip_ckpt, clip_cache, phase='test'),
            BOLD5000_CLIP_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform, num_voxels, clip_name, clip_ckpt, clip_cache, phase='test'))


class BOLD5000_CLIP_dataset(torch.utils.data.Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity, num_voxels=0, 
                 clip_name='', clip_ckpt='', clip_cache='', phase='train'):
        self.fmri = fmri
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        _, train_preprocess, test_preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=clip_ckpt)
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.image_preprocess = train_preprocess if phase == 'train' else test_preprocess
        self.num_voxels = num_voxels

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        fmri = self.fmri[index]

        img = Image.fromarray(self.image[index])
        fmri = np.expand_dims(fmri, axis=0)

        return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img), 
                'image4clip': self.image_preprocess(img), 
                'fmri_target': torch.FloatTensor(fmri)}

    def switch_sub_view(self, sub, subs):
        # Not implemented
        pass

    def create_iterator(self, batch_size):
        sample_loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=8,
            drop_last=False,
            shuffle=True
        )

        for item in sample_loader:
            yield item

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                if k in ['fmri', 'image', 'image4clip', 'label', 'text']:
                    res[k] = default_collate(temp_)
                else:
                    res[k] = temp_
            else:
                res[k] = None
        return res


class CLIP_METRIC_dataset(torch.utils.data.Dataset):
    def __init__(self, images):
        # images: [ALL, N+1, 3, H, W]
        B, N, C, H, W = images.shape
        self.data = []
        for i in range(B):
            gt = rearrange(images[i,0], 'c h w -> h w c')
            for j in range(1, N):
                pred = rearrange(images[i,j], 'c h w -> h w c')
                self.data.append((pred, gt))

        _, _, self.preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', 
                                                                      pretrained='frozen_laion5b_s13b_b90k')
      
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pred, gt = self.data[index]

        pred = Image.fromarray(pred)
        gt = Image.fromarray(gt)

        return {'pred_img': self.preprocess(pred), 'gt_img': self.preprocess(gt)}
