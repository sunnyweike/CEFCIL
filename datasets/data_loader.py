import math
import os, glob
import random
from itertools import compress

import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import SVHN as TorchVisionSVHN

from . import base_dataset as basedat
from . import memory_dataset as memd
from .dataset_config import dataset_config
from .autoaugment import CIFAR10Policy, ImageNetPolicy
from .ops import Cutout


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, num_workers, pin_memory, validation=.1,extra_aug="",num_experts=5, random_seed=0,inc_sub_rate=0.5, collapse_alpha=0, nc_inc_tasks=5, excelpath="", nc_total=None):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load, ini_trn_load, random_ini_trn_load, inc_trn_load = [], [], [],[],[],[]
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      test_resize=dc['test_resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'],
                                                      extra_aug=extra_aug, ds_name=cur_dataset)

        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla, ini_trn_dset, inc_trn_dest = get_datasets(
            cur_dataset, dc['path'], num_tasks, nc_first_task, validation=validation, trn_transform=trn_transform,
            tst_transform=tst_transform, class_order=dc['class_order'], num_experts=num_experts,
            random_seed=random_seed, inc_sub_rate=inc_sub_rate, collapse_alpha=collapse_alpha, nc_inc_tasks=nc_inc_tasks, excelpath=excelpath, nc_total=nc_total)

        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]


        # extend final taskcla list
        taskcla.extend(curtaskcla)
        # taskcla=[0,50]
        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))

        for e in range(num_experts):
            ini_trn_load.append(data.DataLoader(ini_trn_dset[e], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory))
        for t in range(num_tasks):
            for e in range(num_experts):
                # data_inc_trn_load[tt][sub].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
                inc_trn_load.append(data.DataLoader(inc_trn_dest[t*num_experts+e], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory))

    return trn_load, val_load, tst_load, taskcla, ini_trn_load, inc_trn_load


def get_datasets(dataset, path, num_tasks, nc_first_task, validation, trn_transform, tst_transform, class_order=None,num_experts=5, random_seed=0, inc_sub_rate=0.5, collapse_alpha=0, nc_inc_tasks=5, excelpath="", nc_total=None):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset,ini_trn_dset, inc_trn_dest = [], [], [], [], []
    random_ini_trn_dset = {}
    data_inc_trn = {}
    for aa in range(num_experts):
        data_inc_trn[aa] = {}
        for bb in range(num_tasks):
            data_inc_trn[aa][bb] = {'x': [], 'y': []}
    for aa in range(num_experts):
        random_ini_trn_dset[aa] = {'x': [], 'y': []}
    new_random_ini_trn_dset = {}
    for aa in range(num_experts):
        new_random_ini_trn_dset[aa] = {'x': [], 'y': []}

    if 'mnist' in dataset:
        tvmnist_trn = TorchVisionMNIST(path, train=True, download=True)
        tvmnist_tst = TorchVisionMNIST(path, train=False, download=True)
        trn_data = {'x': tvmnist_trn.data.numpy(), 'y': tvmnist_trn.targets.tolist()}
        tst_data = {'x': tvmnist_tst.data.numpy(), 'y': tvmnist_tst.targets.tolist()}
        # compute splits
        all_data, taskcla, class_indices, data_ini_trn, data_inc_trn = memd.get_data(trn_data, tst_data, validation=validation,
                                                                                 num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                                 shuffle_classes=class_order is None, class_order=class_order,
                                                                                 inc_sub_rate=inc_sub_rate, excelpath=excelpath)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, taskcla, class_indices, data_ini_trn, data_inc_trn = memd.get_data(trn_data, tst_data, validation=validation,
                                                                                 num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                                 shuffle_classes=class_order is None, class_order=class_order,num_experts=num_experts,
                                                                                 random_seed=random_seed,inc_sub_rate=inc_sub_rate, excelpath=excelpath)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'svhn':
        tvsvhn_trn = TorchVisionSVHN(path, split='train', download=True)
        tvsvhn_tst = TorchVisionSVHN(path, split='test', download=True)
        trn_data = {'x': tvsvhn_trn.data.transpose(0, 2, 3, 1), 'y': tvsvhn_trn.labels}
        tst_data = {'x': tvsvhn_tst.data.transpose(0, 2, 3, 1), 'y': tvsvhn_tst.labels}
        # Notice that SVHN in Torchvision has an extra training set in case needed
        # tvsvhn_xtr = TorchVisionSVHN(path, split='extra', download=True)
        # xtr_data = {'x': tvsvhn_xtr.data.transpose(0, 2, 3, 1), 'y': tvsvhn_xtr.labels}

        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order, excelpath=excelpath)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'imagenet_32' in dataset:
        import pickle
        # load data
        x_trn, y_trn = [], []
        for i in range(1, 11):
            with open(os.path.join(path, 'train_data_batch_{}'.format(i)), 'rb') as f:
                d = pickle.load(f)
            x_trn.append(d['data'])
            y_trn.append(np.array(d['labels']) - 1)  # labels from 0 to 999
        with open(os.path.join(path, 'val_data'), 'rb') as f:
            d = pickle.load(f)
        x_trn.append(d['data'])
        y_tst = np.array(d['labels']) - 1  # labels from 0 to 999
        # reshape data
        for i, d in enumerate(x_trn, 0):
            x_trn[i] = d.reshape(d.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x_tst = x_trn[-1]
        x_trn = np.vstack(x_trn[:-1])
        y_trn = np.concatenate(y_trn)
        trn_data = {'x': x_trn, 'y': y_trn}
        tst_data = {'x': x_tst, 'y': y_tst}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order, excelpath=excelpath)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'imagenet_subset_kaggle':
        _ensure_imagenet_subset_prepared(path)
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, taskcla, class_indices, data_ini_trn, data_inc_trn  = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                validation=validation, shuffle_classes=class_order is None,
                                                                class_order=class_order, num_experts=num_experts,
                                                                random_seed=random_seed, inc_sub_rate=inc_sub_rate,
                                                                nc_inc_tasks=nc_inc_tasks, excelpath=excelpath, nc_total=nc_total)
        Dataset = basedat.BaseDataset
    elif dataset == 'tiny_imagenet_200':
        _ensure_tiny_imagenet_200_prepared(path)
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, taskcla, class_indices, data_ini_trn, data_inc_trn  = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                validation=validation, shuffle_classes=class_order is None,
                                                                class_order=class_order, num_experts=num_experts,
                                                                random_seed=random_seed, inc_sub_rate=inc_sub_rate,
                                                                nc_inc_tasks=nc_inc_tasks, excelpath=excelpath, nc_total=nc_total)
        Dataset = basedat.BaseDataset

    elif dataset == 'domainnet':
        ##classes_per_domain=5, num_tasks=36

        if nc_total is None:
            if nc_inc_tasks==10:
                numclasses = 240
            elif nc_inc_tasks==25:
                numclasses = 300
            else:
                numclasses = 180
        else:
            numclasses = nc_total
        _ensure_domainnet_prepared(path, classes_per_domain=nc_inc_tasks, num_classes=numclasses)

        all_data, taskcla, class_indices, data_ini_trn, data_inc_trn = basedat.get_data(path, num_tasks=num_tasks,
                                                                nc_first_task=nc_first_task,validation=validation,
                                                                shuffle_classes=False, class_order=None,
                                                                num_experts=num_experts, random_seed=random_seed,
                                                                inc_sub_rate=inc_sub_rate, nc_inc_tasks=nc_inc_tasks,
                                                                excelpath=excelpath, nc_total=numclasses)
        Dataset = basedat.BaseDataset

    # get datasets, apply correct label offsets for each task
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices))

    #将初始阶段的所有数据进行随机打乱，并将data_ini_trn子集改造成data_ini_loader
    np.random.seed(random_seed)
    for e in range(num_experts):
        for index in range(nc_first_task):
            aaa = data_ini_trn[e][index]['x']
            for id in range(len(aaa)):
                random_ini_trn_dset[e]['x'].append(data_ini_trn[e][index]['x'][id])
                random_ini_trn_dset[e]['y'].append(data_ini_trn[e][index]['y'][id])

        random_ini_trn_dset[e]['x'] = np.asarray(random_ini_trn_dset[e]['x'])
        random_ini_trn_dset[e]['y'] = np.asarray(random_ini_trn_dset[e]['y'])
        random_data_id = list(range(len(random_ini_trn_dset[e]['x'])))
        random.shuffle(random_data_id)  #shuffle() 方法是直接对原列表进行操作！操作后会直接改变原列表！
        random_data_id = random.sample(random_data_id, int(np.round(len(random_data_id))))
        new_random_ini_trn_dset[e]['x'] = random_ini_trn_dset[e]['x'][random_data_id]
        new_random_ini_trn_dset[e]['y'] = random_ini_trn_dset[e]['y'][random_data_id]

        if collapse_alpha == 0:
            ini_trn_dset.append(Dataset(random_ini_trn_dset[e], trn_transform, class_indices))
            data_inc_trn[0][e]['x'] = np.asarray(random_ini_trn_dset[e]['x'])
            data_inc_trn[0][e]['y'] = random_ini_trn_dset[e]['y']
        else:
            ini_trn_dset.append(Dataset(new_random_ini_trn_dset[e], trn_transform, class_indices))
            data_inc_trn[0][e]['x'] = np.asarray(new_random_ini_trn_dset[e]['x'])
            data_inc_trn[0][e]['y'] = new_random_ini_trn_dset[e]['y']
            
        inc_trn_dest.append(Dataset(data_inc_trn[0][e], trn_transform, class_indices))

    for task in range(1, num_tasks):
        for e in range(num_experts):
            data_inc_trn[task][e]['x'] = np.asarray(data_inc_trn[task][e]['x'])
            inc_trn_dest.append(Dataset(data_inc_trn[task][e], trn_transform, class_indices))

    return trn_dset, val_dset, tst_dset, taskcla, ini_trn_dset, inc_trn_dest


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


def get_transforms(resize, test_resize, pad, crop, flip, normalize, extend_channel, extra_aug="", ds_name=""):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []
    
    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # test only resize
    if test_resize is not None:
        tst_transform_list.append(transforms.Resize(test_resize))

    # crop
    if crop is not None:
        if 'cifar' in ds_name.lower():
            trn_transform_list.append(transforms.RandomCrop(crop))
        else:
            trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    trn_transform_list.append(transforms.ColorJitter(brightness=63 / 255))
    if extra_aug == 'fetril':  # Similar as in PyCIL
        if 'cifar' in ds_name.lower():
            trn_transform_list.append(CIFAR10Policy())
        elif 'imagenet' in ds_name.lower(): ##含tiny_imagenet_200
            trn_transform_list.append(ImageNetPolicy())
        elif 'domainnet' in ds_name.lower():
            trn_transform_list.append(ImageNetPolicy())
        else:
            raise RuntimeError(f'Please check and update the data agumentation code for your dataset: {ds_name}')
      
    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())
    
    if extra_aug == 'fetril':  # Similar as in PyCIL
        trn_transform_list.append(Cutout(n_holes=1, length=16))
   
    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)


def _ensure_imagenet_subset_prepared(path):
    assert os.path.exists(path), f"Please first download and extract dataset from: https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn to dir: {path}"
    ds_conf = dataset_config['imagenet_subset_kaggle']
    clsss2idx = {c:i for i, c in enumerate(ds_conf['lbl_order'])}
    print(f'Generating train/test splits for ImageNet-Subset directory: {path}')
    def prepare_split(split='train', outfile='train_100.txt'):
        with open(f"{path}/{outfile}", 'wt') as f:
            for fn in glob.glob(f"{path}/data/{split}/*/*"):
                #c = fn.split('\\')[-2]   ####单机调试时用'\\'
                c = fn.split('/')[-2]     ####云平台运行时用'/'
                lbl = clsss2idx[c]
                relative_path = fn.replace(f"{path}/", '')
                f.write(f"{relative_path} {lbl}\n")
    prepare_split()
    prepare_split('val', outfile='test_100.txt')


def _ensure_tiny_imagenet_200_prepared(path):
    assert os.path.exists(path), f"Please first download and extract dataset tiny_imagenet_200 to dir: {path}"
    ds_conf = dataset_config['tiny_imagenet_200']
    clsss2idx = {c:i for i, c in enumerate(ds_conf['lbl_order'])}
    print(f'Generating train/test splits for tiny_imagenet_200 directory: {path}')
    def prepare_split(split='train', outfile='train_200.txt'):
        with open(f"{path}/{outfile}", 'wt') as f:
            for fn in glob.glob(f"{path}/data/{split}/*/*/*"):
                #c = fn.split('\\')[-3]   ####单机调试时用'\\'
                c = fn.split('/')[-3]     ####云平台运行时用'/'
                lbl = clsss2idx[c]
                relative_path = fn.replace(f"{path}/", '')
                f.write(f"{relative_path} {lbl}\n")
    prepare_split()
    prepare_split('val', outfile='test_200.txt')



def _ensure_domainnet_prepared(path, classes_per_domain=5, num_classes=180):
    assert os.path.exists(path), f"Please first download and extract dataset from: http://ai.bu.edu/M3SDA/#dataset into:{path}"
    domains = ["clipart", "infograph", "painting",  "quickdraw", "real", "sketch"] * (num_classes //(classes_per_domain*6) )
    for set_type in ["train", "test"]:
        samples = []
        for i, domain in enumerate(domains):
            with open(f"{path}/{domain}_{set_type}.txt", 'r') as f:
                lines = list(map(lambda x: x.replace("\n", "").split(" "), f.readlines()))
            paths, classes = zip(*lines)
            classes = np.array(list(map(float, classes)))
            offset = classes_per_domain * i
            for c in range(classes_per_domain):
                is_class = classes == c + ((i // 6) * classes_per_domain)
                class_samples = list(compress(paths, is_class))
                samples.extend([*[f"{row} {c + offset}" for row in class_samples]])
        with open(f"{path}/{set_type}_{num_classes}.txt", 'wt') as f:
            for sample in samples:
                f.write(f"{sample}\n")
