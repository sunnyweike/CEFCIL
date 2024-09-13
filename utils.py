import os
import torch
import random
import numpy as np


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        avgs = []
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    avg = 100 * metric[i, :i].mean()
            else:
                avg = 100 * metric[i, :i + 1].mean()
            print('\tAvg.:{:5.1f}% \n'.format(avg), end='')
            avgs.append(avg)
        if "Acc" in name:
            print('Average incremental:{:5.1f}% \n'.format(np.mean(avgs)), end='')
            ##新比较指标聚焦于在CIL任务上对新增类的Tag Acc:只在增量学习任务上（不含初始任务）对所有已学的新增类（不含初始任务上的基本类）的Tag Acc按增量任务求平均 Avg Tag ACC of CIL tasks
            print('-' * 50)
            print(name+' just on CIL taskes\n')
            avgsCIL = []
            for i in range(1, metric.shape[0]):
                print('\t', end='')
                for j in range(1, metric.shape[1]):
                    print('{:5.1f}% '.format(100 * metric[i, j]), end='')
                if np.trace(metric) == 0.0:
                    if i > 0:
                        avg = 100 * metric[i, 1:i].mean()
                else:
                    avg = 100 * metric[i, 1:i + 1].mean()
                print('\tAvg.:{:5.1f}% \n'.format(avg), end='')
                avgsCIL.append(avg)
            print('Average just on CIL taskes:{:5.1f}% \n'.format(np.mean(avgsCIL)), end='')

    print('*' * 108)
