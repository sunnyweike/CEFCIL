import copy
import random

import numpy as np
import torch
import os
import torch.nn.functional as F

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal
from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .incremental_learning import Inc_Learning_Appr
from .loss_DimensionCollapse import DimensionCollapse
torch.backends.cuda.matmul.allow_tf32 = False

from .gmm import GaussianMixture

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
EPSILON = 1e-8

def softmax_temperature(x, dim, tau=1.0):
    return torch.softmax(x / tau, dim=dim)


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()




class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=2, ftepochs=1, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, ftwd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, max_experts=999, alpha=1.0, tau=3.0, shared=0, use_nmc=False, num_ini_classes=10,num_experts=5,num_tasks=20,random_seed=0,datasets=['cifar100_icarl'],collapse_alpha=0,inc_sub_rate=0.5,ini_kd_loss=0.1,
                 initialization_strategy="first"):
        super(Appr, self).__init__(model, device, nepochs, ftepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train,logger,num_ini_classes,num_experts,num_tasks,random_seed,datasets,collapse_alpha,inc_sub_rate,ini_kd_loss
                                   )
        self.max_experts = max_experts
        self.model.bbs = self.model.bbs[:num_experts]
        self.alpha = alpha
        self.tau = tau
        self.patience = patience
        self.use_nmc = use_nmc
        self.ftwd = ftwd
        self.model.to(device)
        self.experts_distributions = []
        self.ini_experts_distributions=[]
        self.ini_kd_loss=ini_kd_loss
        self.task_offset=[0]
        self.shared_layers = []
        self.count_tr=[]
        self.ini_model=[]

        self._init_protos = []
        self._protos = []
        self._cov_mat_shrink = []
        self._norm_cov_mat = []
        self._cov_mat = []
        if inc_sub_rate > 0:
            self.incLearn = True
        else:
            self.incLearn = False
        if inc_sub_rate <= 1/num_experts:
            self.needChoose = False
        else:
            self.needChoose = True

        if shared > 0:
            self.shared_layers = ["conv1_starting.weight", "bn1_starting.weight", "bn1_starting.bias", "layer1"]
            if shared > 1:
                self.shared_layers.append("layer2")
                if shared > 2:
                    self.shared_layers.append("layer3")
                    if shared > 3:
                        self.shared_layers.append("layer4")

        self.initialization_strategy = initialization_strategy

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--max-experts',
                            help='Maximum number of experts',
                            type=int,
                            default=5)
        parser.add_argument('--initialization-strategy',
                            help='How to initialize experts weight',
                            type=str,
                            choices=["first", "random"],
                            default="first")
        parser.add_argument('--ftwd',
                            help='Weight decay for finetuning',
                            type=float,
                            default=0)

        parser.add_argument('--use_nmc',
                            help='Use nearest mean classifier instead of bayes',
                            action='store_true',
                            default=False)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.9)
        parser.add_argument('--tau',
                            help='softmax temperature',
                            type=float,
                            default=3.0)
        parser.add_argument('--ini_kd_loss',
                            help='根蒸馏损失',
                            type=float,
                            default=0.1)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader, ini_trn_load, inc_trn_loader):
        transforms = val_loader[t].dataset.transform
        self.task_offset.append(self.task_offset[t] + self.model.taskcla[t][1])
        if t == 0:
            for e in range(self.num_experts):
                print(f"Training backbone_{e} on task_{t}:")
                self.train_backbone(e, ini_trn_load)
                self.ini_model.append(copy.deepcopy(self.model.bbs[e]))
                self._protos.append([])
                self._cov_mat.append([])
                self._cov_mat_shrink.append([])
                self._norm_cov_mat.append([])
        elif self.incLearn:
            expert_to_finetune = self._choose_model_to_finetune(t, trn_loader[t], transforms)############
            print(f"Finetuning backbone_{expert_to_finetune} on task_{t}:")
            self.finetune_backbone(t, expert_to_finetune, inc_trn_loader[t*self.num_experts])
            self.finetune_other_backbone(t, expert_to_finetune, inc_trn_loader)

        print(f"Creating distributions for task_{t}:")
        self.create_distributions(t, trn_loader[t], transforms)

    def train_backbone(self, e, ini_trn_load):
        if self.initialization_strategy == "random" or e==0:
            self.model.bbs.append(self.model.bb_fun(num_classes=self.num_ini_classes, num_features=self.model.num_features))
        else:
            self.model.bbs.append(copy.deepcopy(self.model.bbs[0]))
        # self.model.bbs.append(self.model.bb_fun(num_classes=self.num_ini_classes, num_features=self.model.num_features))
        model = self.model.bbs[e]
        # model.fc = nn.Linear(self.model.num_features, self.model.taskcla[e][1])
        model.fc = nn.Linear(self.model.num_features, self.num_ini_classes)
        if e == 0:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                for layer_not_to_train in self.shared_layers:
                    if layer_not_to_train in name:
                        model.get_parameter(name).data = self.model.bbs[0].get_parameter(name).data
                        param.requires_grad = False

        print(f'The expert_{e} has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert_{e} has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(e, self.wd)

        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in ini_trn_load[e]:
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out,features = model(images, return_features=True)
                loss = nn.functional.cross_entropy(out, targets.long(), label_smoothing=0.0)
                # 防范维度坍塌
                dimension_collapse = DimensionCollapse()
                loss_dimension_collapse = dimension_collapse(features)
                # loss=(1-self.collapse_alpha)*loss+self.collapse_alpha*loss_dimension_collapse
                loss=loss+self.collapse_alpha * loss_dimension_collapse
                #
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()


        model.fc = nn.Identity()
        self.model.bbs[e] = model
        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")

######################
    @torch.no_grad()
    def _choose_model_to_finetune(self, t, trn_loader, transforms):
        if (self.num_experts == 1) or (not self.needChoose):
            return 0
        self.create_distributions(t, trn_loader, transforms)
        expert_KL = torch.zeros(self.num_experts, device=self.device)
        for bb_num in range(self.num_experts):
            classes_in_t = self.model.taskcla[t][1]
            classes_mean = self._protos[bb_num][-classes_in_t:]  # [-classes_in_t:]的意思是倒着选第“classes_in_t”个元素
            norm_cov_mat = self._norm_cov_mat[bb_num][-classes_in_t:]
            kl_matrix = torch.zeros((len(classes_mean), len(classes_mean)), device=self.device)
            for o, class_mean in enumerate(classes_mean):
                old_gauss = MultivariateNormal(class_mean.to(self.device), covariance_matrix=norm_cov_mat[o].to(self.device))
                for n, class_mean_n in enumerate(classes_mean):
                    new_gauss = MultivariateNormal(class_mean_n.to(self.device), covariance_matrix=norm_cov_mat[n].to(self.device))
                    kl_matrix[n, o] = torch.distributions.kl_divergence(new_gauss, old_gauss).to(self.device)
            expert_KL[bb_num] = torch.mean(kl_matrix)
            self._protos[bb_num] = self._protos[bb_num][:-classes_in_t]
            self._norm_cov_mat[bb_num] = self._norm_cov_mat[bb_num][:-classes_in_t]
            self._cov_mat[bb_num] = self._cov_mat[bb_num][:-classes_in_t]
        expert_to_finetune = int(torch.argmax(expert_KL))
        print(f'expert_KL: {expert_KL}')
        return expert_to_finetune


#################################
    def finetune_backbone(self, t, bb_to_finetune,data_inc_trn_loader):
        if data_inc_trn_loader==[]:
            pass
        else:
            old_model = copy.deepcopy(self.model.bbs[bb_to_finetune])
            for name, param in old_model.named_parameters():
                param.requires_grad = False
            old_model.eval()
            init_model = copy.deepcopy(self.ini_model[bb_to_finetune])
            for name, param in init_model.named_parameters():
                param.requires_grad = False
            init_model.eval()
            model = self.model.bbs[bb_to_finetune]
            for name, param in model.named_parameters():
                param.requires_grad = True
                for layer_not_to_train in self.shared_layers:
                    if layer_not_to_train in name:
                        param.requires_grad = False
            model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])
            model.to(self.device)
            # print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
            # print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

            optimizer, lr_scheduler = self._get_optimizer(bb_to_finetune, wd=self.ftwd, milestones=[30, 60, 80])
            for epoch in range(self.ftepochs):
                train_loss, valid_loss = [], []
                train_hits, val_hits = 0, 0
                drift_n = 0
                optimizer_step_flag = False
                model.train()
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                batch_id=0
                for images, targets in data_inc_trn_loader:
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        old_features = old_model(images)  # resnet with fc as identity returns features by default
                        init_features = init_model(images)  # resnet with fc as identity returns features by default
                    out, features = model(images, return_features=True)
                    loss = self.criterion(t, out, targets.long(), features, old_features, init_features)
####                    print(f"loss on batch {batch_id}: {loss} ")
                    batch_id +=1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                    optimizer.step()
                    optimizer_step_flag = True
                    train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    train_loss.append(float(bsz * loss))
                if optimizer_step_flag:
                    lr_scheduler.step()
            model.fc = nn.Identity()
            self.model.bbs[bb_to_finetune] = model
            # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
            return old_model


    def finetune_other_backbone(self, t, expert_to_finetune, data_inc_trn_loader):
        experts_list = list(range(self.num_experts))
        experts_list.remove(expert_to_finetune)
        i=0
        for j in experts_list:
            print(f"Finetuning other backbones_{j} on task_{t}:")
            i +=1
            self.finetune_backbone(t, j, data_inc_trn_loader[t * self.num_experts + i])

    def criterion(self, t, outputs, targets, features=None, old_features=None, init_features=None):
        """Returns the loss value"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        if old_features is not None:  # Knowledge distillation loss on features
            kd_loss = nn.functional.mse_loss(features, old_features)
            if init_features is not None:  # Knowledge distillation loss on features
                kd_loss =(1-self.ini_kd_loss) * kd_loss + self.ini_kd_loss * nn.functional.mse_loss(features, init_features)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss
        return ce_loss

    def _get_optimizer(self, num, wd, milestones=[60, 120, 160]):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(self.model.bbs[num].parameters(), lr=self.lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def _tukeys_transform(self, x):
        return x
        #beta = self.args["beta"] ##################################################
        # beta=0.5
        # x = torch.tensor(x)
        # if beta == 0:
        #     return torch.log(x)
        # else:
        #     return torch.pow(x, beta)

    def shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0])
        cov_ = cov + (diag_mean * iden) + (off_diag_mean * (1 - iden))
        return cov_

    @torch.no_grad()
    def _create_distributions(self, t, trn_loader, transforms, classes):
        for c in range(classes):
            c = c + self.task_offset[t]
            train_indices = torch.tensor(trn_loader.dataset.labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            for bb_num in range(self.num_experts):
                model = self.model.bbs[bb_num]
                vectors = []
                for images in loader:
                    images = images.to(self.device)
                    features = model(images)
                    _vectors = tensor2numpy(features)
                    vectors.append(_vectors)

                vects=np.concatenate(vectors)
                class_mean = np.mean(vects, axis=0)
                self._protos[bb_num].append(torch.tensor(class_mean).to(self.device))

                vects = self._tukeys_transform(vects)
                cov = torch.tensor(np.cov(vects.T))
                cov = self.shrink_cov(cov)
                self._cov_mat[bb_num].append(cov)
                sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
                cov = cov / (torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
                self._norm_cov_mat[bb_num].append(cov.to(self.device))

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, transf):
        """ Create distributions for task t"""
        self.model.eval()
        classes = self.model.taskcla[t][1]
        self._create_distributions(t, trn_loader, transf, classes)

    @torch.no_grad()
    def _extract_vectors(self, loader):
        vectors = [[] for i in range(self.num_experts)]
        vects = [None] * self.num_experts
        targets =[]
        for images, _targets in loader:
            _targets = _targets.numpy()
            targets.append(_targets)
            for e in range(self.num_experts):
                model = self.model.bbs[e]
                model.eval()
                features = model(images.to(self.device))
                _vectors = tensor2numpy(features)
                vectors[e].append(_vectors)

        for e in range(self.num_experts):
            vects[e] = np.concatenate(vectors[e])
            vects[e] = (vects[e].T / (np.linalg.norm(vects[e].T, axis=0) + EPSILON)).T
        return vects, np.concatenate(targets)

    def _mahalanobis(self, vector, class_means, cov):
        class_means = self._tukeys_transform(class_means)
        x_minus_mu = F.normalize(vector, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if self.use_nmc:    # Use nearest mean classifier instead of bayes
            cov = torch.eye(cov.shape[0])  # identity covariance matrix for euclidean distance
        inv_covmat = torch.linalg.pinv(cov).float().to(self.device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()

    def _maha_dist(self, t, vectors):
        maha_dist = [[] for i in range(self.num_experts)]
        for e in range(self.num_experts):
            vector = torch.tensor(vectors[e]).to(self.device)
            vector = self._tukeys_transform(vector)
            known_classes = self.task_offset[t]
            total_classes = self.task_offset[t + 1]
            for class_index in range(total_classes):
                dist = self._mahalanobis(vector, self._protos[e][class_index], self._norm_cov_mat[e][class_index])
                maha_dist[e].append(dist)
            maha_dist[e] = np.array(maha_dist[e])  # [total_classes, N]
        return maha_dist  # [self.num_experts, total_classes, N]

    @torch.no_grad()
    def eval(self, t, tst_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag= 0, 0, 0
        vectors, y_true = self._extract_vectors(tst_loader)
        dists = self._maha_dist(t, vectors)  # [self.num_experts, nb_classes, N]
        dist = dists[0].copy()
        for e in range(1, self.num_experts):
            dist += dists[e]
        scores = dist.T  # [N, total_classes], choose the one with the smallest distance
        known_classes = self.task_offset[t]
        total_classes = self.task_offset[t + 1]
        scores_taw = scores[:, known_classes:total_classes]
        y_pred = np.argsort(scores, axis=1)[:, 0]  # [N, 1]
        total_acc_tag = np.around( (y_pred == y_true).sum()/ len(y_true), decimals=2)

        y_pred_taw = np.argsort(scores_taw, axis=1)[:, 0] + known_classes # [N, 1]
        total_acc_taw =  np.around( (y_pred_taw == y_true).sum()  / len(y_true), decimals=2)

        return total_loss, total_acc_taw, total_acc_tag

