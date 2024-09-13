import os
import time
import torch
import argparse
import importlib
import numpy as np
import random
import utils

import approach

from functools import reduce
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var


def seed_everything(seed=1):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default='exp_imagenet_0804-ma' , type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args  default=cifar100_icarl
    parser.add_argument('--datasets', default=['tiny_imagenet_200'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=128, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=6, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=100, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--nc-inc-tasks', default=20, type=int, required=False,
                        help='Number of classes in per incremental learning task (default=%(default)s)')
    parser.add_argument('--nc-total', default=200, type=int, required=False,
                        help='Number of all classes in the train dataset (default=%(default)s)')

    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet18', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='ILFTF', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=2, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--ftepochs', help='Number of epochs for finetuning an expert',
                        type=int,    default=1, required=False)
    parser.add_argument('--lr', default=0.05, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=1, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=5e-4, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    parser.add_argument('--use-test-as-val', action='store_true',
                        help='Use test set as a validation set, common practice in other benchmarks')
    parser.add_argument('--extra-aug', default='', type=str, choices=['', 'fetril'],
                        help='Additional data augmentations (default=%(default)s)')

    parser.add_argument('--small_data_rate', default=0.1, type=float,
                        help='small_data_rate')
    parser.add_argument('--num_experts', default=2, type=int,
                        help='num_experts')
    parser.add_argument('--random_seed', default=1993, type=int,
                        help='random_seed')
    parser.add_argument('--collapse_alpha', default=0.2, type=float,
                        help='collapse_alpha')
    parser.add_argument('--inc_sub_rate', default=0.5, type=float,
                        help='inc_sub_rate')
    parser.add_argument('--shared', default=1, type=float,
                        help='Number of shared blocks')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    if args.num_experts == 1 and args.inc_sub_rate > 0:
        args.inc_sub_rate=1.0


    base_kwargs = dict(nepochs=args.nepochs, ftepochs=args.ftepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train, num_ini_classes=args.nc_first_task, num_experts=args.num_experts,
                       random_seed=args.random_seed, num_tasks=args.num_tasks, collapse_alpha=args.collapse_alpha, inc_sub_rate=args.inc_sub_rate, shared=args.shared)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    from networks.network import LLL_Net, ExtractorEnsemble
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    #print(appr_args.max_experts)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    #assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla, ini_trn_loader, inc_trn_loader = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory,
                                                              validation=0.0 if args.use_test_as_val else 0.1,
                                                              extra_aug = args.extra_aug,
                                                              num_experts=args.num_experts,
                                                              random_seed=args.random_seed,
                                                              inc_sub_rate=args.inc_sub_rate,
                                                              collapse_alpha= args.collapse_alpha,
                                                              nc_inc_tasks = args.nc_inc_tasks,
                                                              excelpath=args.exp_name,
                                                              nc_total=args.nc_total)
    # print(data_trn)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    if args.use_test_as_val:
        val_loader = tst_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    seed_everything(seed=args.seed)
    if args.approach == "ILFTF":
        net = ExtractorEnsemble(init_model, taskcla, args.network, device)
    else:
        net = LLL_Net(init_model, taskcla, remove_existing_head=not args.keep_existing_head)
    seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)

    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task)) # Task-Aware（任务可知的） Multi-Head条件下的精确度
    acc_tag = np.zeros((max_task, max_task)) # Task-Agnostic（任务无偏的、任务不可知的） Multi-Head条件下的精确度
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))

    for t in range(args.num_tasks):
        # Early stop tasks if flag
        if t >= max_task:
            continue
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # experts-Train
        appr.train(t, trn_loader, val_loader, ini_trn_loader, inc_trn_loader)

        #if t >= 0:  # increment-Train
        print('-' * 108)
        for u in range(t+1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])
            # Save
            print('Save at ' + os.path.join(args.results_path, full_exp_name))
            logger.log_result(acc_taw, name="acc_taw", step=t)
            logger.log_result(acc_tag, name="acc_tag", step=t)
            logger.log_result(forg_taw, name="forg_taw", step=t)
            logger.log_result(forg_tag, name="forg_tag", step=t)
            logger.save_model(net.state_dict(), task=t)
            logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
            logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
            aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
            logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
            logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)
            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)
        if t==0:
            print('[Elapsed time in init_stage = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
            tstart = time.time()
    # Print Summary
    if appr_args.use_nmc:
        print('**********************Using NMC**********************')
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time for incremental learning = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
