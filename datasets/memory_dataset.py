import random
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import math


class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.fromarray(self.images[index])
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(trn_data, tst_data, num_tasks, nc_first_task=10, validation=0.0, shuffle_classes=True, class_order=None,num_experts=5,random_seed=0,inc_sub_rate=0.5, excelpath=""):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    data_ini_trn={}
    data_inc_trn = {}
    taskcla = []
    temp_data={}

    #num_class=nc_first_task+ int((num_tasks-1) * nc_inc_tasks )
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)
    # compute classes per task and num_tasks
    assert nc_first_task < num_classes, "first task wants more classes than exist"
    remaining_classes = num_classes - nc_first_task
    nc_inc_tasks= remaining_classes // (num_tasks - 1)

    cpertask = []
    cpertask.append(nc_first_task)
    for i in range(num_tasks-1):
        cpertask.append(nc_inc_tasks)
    cpertask[num_tasks-1] = remaining_classes - nc_inc_tasks*(num_tasks - 2)
    cpertask = np.asarray(cpertask)
    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"

    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for t in range(num_tasks):
        data[t] = {}
        data[t]['name'] = 'task-' + str(t)
        data[t]['trn'] = {'x': [], 'y': []}
        data[t]['val'] = {'x': [], 'y': []}
        data[t]['tst'] = {'x': [], 'y': []}

    #创建初始训练集的空数组
    for e in range(num_experts):
        data_ini_trn[e] = {}
        for c in range(nc_first_task):
            data_ini_trn[e][c] = {'x': [], 'y': []}
    # 创建增量学习训练集的空数组
    for t in range(num_tasks):
        data_inc_trn[t] = {}
        for e in range(num_experts):
            data_inc_trn[t][e] = {'x': [], 'y': []}

    for t in range(num_tasks):
        temp_data[t] = {}
        temp_data[t]['trn'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)#isin()相当于一个过滤器，选出trn_data['y']包含class_order的就为true，反之为false
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)#开始把乱序的图片整理成按照特定类顺序排列，这里先是把原标签换成类序列中对应的下标
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label)
        #data[this_task]['trn']['y'].append(this_label - init_class[this_task])#开始把乱序的图片整理成按照特定类顺序排列，每个任务中的类序从0开始编序

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label)
        #data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for t in range(num_tasks):
        data[t]['ncla'] = len(np.unique(data[t]['trn']['y']))
        assert data[t]['ncla'] == cpertask[t], "something went wrong splitting classes"
        # convert them to numpy arrays
        for split in ['trn', 'val', 'tst']:
            data[t][split]['x'] = np.asarray(data[t][split]['x'])
            data[t][split]['y'] = np.asarray(data[t][split]['y'])

    # validation
    if validation > 0.0:
        for tt in range(num_tasks):
            for cc in range(data[tt]['ncla']):
                index = cc + init_class[tt]
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == index)[0])#numpy.where(condition[,x,y])，返回元素，可以是x或y，具体取决于条件(condition)
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                data[tt]['val']['x'] = data[tt]['trn']['x'][rnd_img]
                data[tt]['val']['y'] = data[tt]['trn']['y'][rnd_img]
                rest_indx = list(set(np.arange(len(data[tt]['trn']['y']))) - set(rnd_img))
                data[tt]['trn']['x'] = data[tt]['trn']['x'][rest_indx]
                data[tt]['trn']['y'] = data[tt]['trn']['y'][rest_indx]


    #创建初始训练子集；并打乱增量学习任务的训练集中的数据顺序;第一个专家的训练集是全数据，其余四个专家将另一份全训练集随机比例分为四份构成小训练集(无放回)
    L = [[[] for i in range(nc_first_task)] for i in range(num_experts)]
    np.random.seed(random_seed)
    for cc in range(data[0]['ncla']):
        index = cc
        cls_idx = list(np.where(np.asarray(data[0]['trn']['y']) == cc)[0])  # numpy.where(condition[,x,y])，返回元素，可以是x或y，具体取决于条件(condition)
        rest_cls_idx = random.sample(cls_idx, int(np.round(len(cls_idx))))
        rest_experts = num_experts - 1
        while (rest_experts >= 1):
            a = np.random.random(rest_experts)
            a /= a.sum()
            L[rest_experts][index] = a[0]/(num_experts-rest_experts)
            rnd_subdata_idx = random.sample(rest_cls_idx, int(np.round(len(rest_cls_idx) * a[0])))
            data_ini_trn[rest_experts][index]['x'] = data[0]['trn']['x'][rnd_subdata_idx]
            data_ini_trn[rest_experts][index]['y'] = data[0]['trn']['y'][rnd_subdata_idx]
            # for ii in range(len(rnd_img_subdata)):
            #     data_ini_trn[rest_experts][index]['x'].append(data[0]['trn']['x'][rnd_img_subdata[ii]])
            #     data_ini_trn[rest_experts][index]['y'].append(index)
            rest_cls_idx = [x for x in rest_cls_idx if x not in rnd_subdata_idx]
            rest_experts -= 1
        L[0][index] = 0.8
        rnd_subdata_idx = random.sample(cls_idx, int(np.round(len(cls_idx)*L[0][index])))
        data_ini_trn[0][index]['x'] = data[0]['trn']['x'][rnd_subdata_idx]
        data_ini_trn[0][index]['y'] = data[0]['trn']['y'][rnd_subdata_idx]
        # for ii in range(len(rnd_img_subdata)):
        #     data_ini_trn[0][index]['x'].append(data[0]['trn']['x'][rnd_img_subdata[ii]])
        #     data_ini_trn[0][index]['y'].append(index)
    # convert them to numpy arrays
    for e in range(num_experts):
        for cc in range(data[0]['ncla']):
            data_ini_trn[e][cc]['x'] = np.asarray(data_ini_trn[e][cc]['x'])

    #打乱原数据集内各个任务里数据的顺序，并将其塞入一个临时数据集（temp_data）中
    for t in range(1, num_tasks):  ##data_inc_trn[0]为定义，它实际上就是 data_ini_trn
        num_data = len(data[t]['trn']['x'])
        my_list = list(range(0, num_data))
        random.shuffle(my_list)
        temp_data[t]['trn']['x'] = data[t]['trn']['x'][my_list]
        temp_data[t]['trn']['y'] = data[t]['trn']['y'][my_list]
      # # 形成增量数据集data_inc_trn
        first_e_num = math.floor(num_data * inc_sub_rate)
        if (num_experts - 1)>0:
            othere_num = math.floor(num_data * ((1 - inc_sub_rate) / (num_experts - 1)))
        else:
            othere_num = 0
        for e in range(num_experts):
            if e ==0:## the first expert
                data_inc_trn[t][e]['x'] = temp_data[t]['trn']['x'][:first_e_num-1]
                data_inc_trn[t][e]['y'] = temp_data[t]['trn']['y'][:first_e_num-1]
                temp_data[t]['trn']['x']= temp_data[t]['trn']['x'][first_e_num:]
                temp_data[t]['trn']['y']= temp_data[t]['trn']['y'][first_e_num:]
            else: ##(num_experts>1)
                data_inc_trn[t][e]['x'] = temp_data[t]['trn']['x'][:othere_num-1]
                data_inc_trn[t][e]['y'] = temp_data[t]['trn']['y'][:othere_num-1]
                temp_data[t]['trn']['x']= temp_data[t]['trn']['x'][othere_num:]
                temp_data[t]['trn']['y']= temp_data[t]['trn']['y'][othere_num:]
            #data_inc_trn[t][e]['x'] = np.asarray(data_ini_trn[t][e]['x'])

    #输出各个类在各个子集里的分割比例到xlsx文件中
    df = pd.DataFrame(L)
    # 保存到本地excel
    df.to_excel(excelpath + "IniSubdataset_CRate.xlsx", index=False)

    # other
    n = 0
    for t in range(num_tasks):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order, data_ini_trn, data_inc_trn
