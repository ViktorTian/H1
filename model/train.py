import os
import time
import yaml
import random
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.distributed

import argparse
from thop import profile

from models.submodules.layers import Conv1x1, Conv3x3, Linear, SpikingMatmul
import models.spikingresformer
from utils.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.utils import RecordDict, GlobalTimer, Timer
from utils.utils import count_convNd, count_linear, count_matmul
from utils.utils import DatasetSplitter, DatasetWarpper, CriterionWarpper, DVStransform, SOPMonitor
from utils.utils import is_main_process, save_on_master, tb_record, accuracy, safe_makedirs
from spikingjelly.activation_based import functional, layer, base
from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model
from models.spikingresformer import DSSAWithSSDP
from collections import defaultdict
import matplotlib.pyplot as plt



def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Training')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='spikingresformer_ti', help='model type')
    parser.add_argument('--dataset', default='ImageNet', help='dataset type')
    parser.add_argument('--augment', type=str, help='data augmentation')
    parser.add_argument('--mixup', type=bool, default=False, help='Mixup')
    parser.add_argument('--cutout', type=bool, default=False, help='Cutout')
    parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')

    parser.add_argument('--print-freq', default=5, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--transfer', type=str, help='transfer from pretrained checkpoint')
    parser.add_argument('--input-size', type=int, nargs='+', default=[])
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBN training')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    num_classes: int,
    dataset_type: str,
    input_size: Tuple[int],
    distributed: bool,
    augment: str,
    mixup: bool,
    cutout: bool,
    label_smoothing: float,
    T: int,
):

    if dataset_type == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                                     download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.0,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.5, cutmix_alpha=0.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.0,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.5, cutmix_alpha=0.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset_train, dataset_test = DatasetSplitter(dataset, 0.9,
                                                      True), DatasetSplitter(dataset, 0.1, False)
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       sampler=test_sampler, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'DVS128Gesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))

        dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T,
                                      split_by='number')
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       sampler=test_sampler, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'ImageNet' or dataset_type == 'ImageNet100':
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        dataset_train = torchvision.datasets.ImageFolder(traindir)
        dataset_test = torchvision.datasets.ImageFolder(valdir)
        augment_args = dict(
            scale=[0.08, 1.0],
            ratio=[3. / 4., 4. / 3.],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.4,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.2, cutmix_alpha=1.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            crop_pct=0.95,
            pin_memory=True,
        )
    else:
        raise ValueError(dataset_type)

    return dataset_train, dataset_test, data_loader_train, data_loader_test


# ------------------------------------------------------------------------------
# train_one_epoch – 两阶段逻辑
#   • warm_phase  (epoch < ssdp.warmup_epochs) —— 完全沿用旧脚本的 BP / LR / Loss
#   • ssdp_phase  (epoch ≥ ssdp.warmup_epochs) —— 启动向量化 SSDP + per-class 统计
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# train_one_epoch – two-phase version (warm-up & ssdp)
# ------------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    print_freq: int,
    factor: int,
    scheduler_per_iter=None,
    scaler: Optional[GradScaler] = None,
    one_hot: Optional[int] = None,
):
    import time, torch
    import torch.nn.functional as F
    from collections import defaultdict
    from spikingjelly.activation_based import functional

    model.train()

    device  = next(model.parameters()).device
    net     = model.module if hasattr(model, 'module') else model
    num_cls = net.classifier.out_features
    # net.current_epoch = epoch
    # -------- 判断阶段 --------
    warm_phase = net.current_epoch < net.ssdp.warmup_epochs

    # -------- 统计 --------
    counts  = torch.zeros(num_cls, device=device)
    correct = torch.zeros(num_cls, device=device)

    class Record:
        def __init__(self, keys):
            self.sum, self.cnt = defaultdict(float), defaultdict(int)
            for k in keys: self.sum[k]=0.0; self.cnt[k]=0
        def update(self,k,v,n=1):
            self.sum[k]+=v*n; self.cnt[k]+=n
        def __getitem__(self,k):
            return self.sum[k]/self.cnt[k] if self.cnt[k] else 0.0
    metrics = Record(['loss','acc@1','acc@5'])

    def accuracy(out,tgt,topk=(1,5)):
        maxk = max(topk); B = tgt.size(0)
        _,pred = out.topk(maxk,1,True,True); pred = pred.t()
        corr = pred.eq(tgt.view(1,-1).expand_as(pred))
        return [corr[:k].reshape(-1).float().sum().mul_(100.0/B) for k in topk]

    t0 = time.time(); model.zero_grad()

    for idx,(images,targets_raw) in enumerate(data_loader_train):
        images       = images.to(device)
        targets_raw  = targets_raw.to(device)

        # ---------- 统一标签 ----------
        if targets_raw.ndim > 1:                     # mixup / cutmix -> soft label
            targets_idx  = targets_raw.argmax(-1)
            targets_soft = targets_raw.float()
        else:                                        # 普通整型标签
            targets_idx  = targets_raw
            targets_soft = F.one_hot(targets_raw, num_cls).float() if one_hot else None

        target_tensor = targets_soft if targets_soft is not None else targets_idx

        # ---------- forward & loss ----------
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss    = criterion(outputs, target_tensor)
        else:
            outputs = model(images)
            loss    = criterion(outputs, target_tensor)

        # ---------- backward ----------
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        model.zero_grad()
        if scheduler_per_iter: scheduler_per_iter.step()

        # ---------- 若网络未写回必需属性，重置继续 ----------
        if not (hasattr(net,'out') and hasattr(net,'t_pre') and hasattr(net,'t_post')):
            functional.reset_net(model); continue

        # ===========================================================
        #                 SSDP 更新（仅 ssdp_phase 写权重）
        # ===========================================================
        pre  = (net.out   > 0).float()
        post = (outputs   > 0).float()
        dt   = (net.t_post.unsqueeze(2) - net.t_pre.unsqueeze(1)).abs()
        delta_w = net.ssdp(pre, post, dt, loss=loss, epoch=net.current_epoch)
        if not warm_phase:
            with torch.no_grad():
                net.classifier.weight.add_(delta_w.detach())

        # ---- DSSA.Wproj ----
        last_stage = net.layers[-1]
        dssa = next((m for m in last_stage if isinstance(m, DSSAWithSSDP)), None)
        if dssa is not None:
            x_in ,x_out = dssa.x_in.detach(), dssa.x_out.detach()   # [T,B,C,H,W]
            T,B,C,_,_   = x_in.shape
            pre_seq  = (x_in .reshape(T,B,C,-1) > 0).any(-1).float()
            post_seq = (x_out.reshape(T,B,C,-1) > 0).any(-1).float()

            pre_first  = pre_seq .cumsum(0).argmax(0).float()
            post_first = post_seq.cumsum(0).argmax(0).float()
            dt2  = (post_first.unsqueeze(2) - pre_first.unsqueeze(1)).abs()

            pre2  = (pre_seq .sum(0) > 0).float()
            post2 = (post_seq.sum(0) > 0).float()

            dw2 = net.ssdp_dssa(pre2, post2, dt2, loss=loss, epoch=net.current_epoch)
            if not warm_phase:
                with torch.no_grad():
                    dssa.Wproj.weight.add_(dw2.detach().view(C, C, 1, 1))

        # ---------- reset neuron states ----------
        functional.reset_net(model)

        # ---------- 记录 ----------
        metrics.update('loss', loss.item(), images.size(0))
        acc1, acc5 = accuracy(outputs, targets_idx)
        metrics.update('acc@1', acc1.item(), images.size(0))
        metrics.update('acc@5', acc5.item(), images.size(0))

        if not warm_phase:  # 统计 per-class 只在 ssdp_phase
            t_flat = targets_idx.view(-1)
            p_flat = outputs.argmax(1).view(-1)
            counts  += torch.bincount(t_flat, minlength=num_cls).float()
            correct += torch.bincount(
                t_flat,
                weights=(p_flat == t_flat).float(),
                minlength=num_cls).float()

        if print_freq and (idx+1) % max(1,len(data_loader_train)//print_freq)==0:
            speed = (idx+1)*images.size(0)*factor/(time.time()-t0)
            logger.debug(f'[{idx+1}/{len(data_loader_train)}] '
                         f'it/s:{speed:.2f}  '
                         f'loss:{metrics["loss"]:.4f}  '
                         f'acc1:{metrics["acc@1"]:.2f}  acc5:{metrics["acc@5"]:.2f}')

    per_class_acc = (100. * correct / counts.clamp(min=1.0)) \
                    if not warm_phase else torch.zeros(num_cls, device=device)

    return metrics['loss'], metrics['acc@1'], metrics['acc@5'], per_class_acc





def evaluate(model, criterion, data_loader, print_freq, logger, one_hot=None):
    model.eval()
    from utils.utils import RecordDict, accuracy
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            if one_hot:
                target = F.one_hot(target, one_hot).float()
            output = model(image) # [B,C]
            loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            if target.dim() > 1:
                target = target.argmax(-1)
            # 同样这里不对output做mean(0)，保持[B,C]
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave

def test(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    input_size: Tuple[int],
    args: argparse.Namespace,
    logger: logging.Logger,
):
    logger.info('[Test]')
    from utils.utils import RecordDict, accuracy, SOPMonitor
    mon = SOPMonitor(model)
    model.eval()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        t = time.time()
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image) # [B,C]
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1,5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if args.print_freq != 0 and ((idx + 1) % int(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))
        logger.info('Throughput: {:.5f} it/s'.format(
            len(data_loader_test) * args.batch_size / (time.time() - t)))
    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    step_mode = 's'
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break

    from utils.utils import count_convNd, count_linear, count_matmul
    ops, params = profile(
        model, inputs=(torch.rand(input_size).cuda().unsqueeze(0), ), verbose=False, custom_ops={
            layer.Conv2d: count_convNd,
            Conv3x3: count_convNd,
            Conv1x1: count_convNd,
            Linear: count_linear,
            SpikingMatmul: count_matmul, })[0:2]
    if step_mode == 'm':
        ops, params = (ops / (1000**3)) / args.T, params / (1000**2)
    else:
        ops, params = (ops / (1000**3)), params / (1000**2)
    functional.reset_net(model)
    logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop
    sops = sops / (1000**3)
    sops = sops / args.batch_size
    if step_mode == 's':
        sops = sops * args.T
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops)))

def cos_annealing_factor(epoch, total_epochs, start_factor, end_factor):
    """
    返回某个[ start_factor -> end_factor ]的余弦退火插值系数
    公式示例: factor = end_factor + 0.5 * (start_factor - end_factor)
                         * (1 + cos(pi * epoch / total_epochs))
    """
    import math
    cos_part = 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    factor = end_factor + (start_factor - end_factor) * cos_part
    return factor

def visualize_attention_snapshots(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epochs: list,
):
    import matplotlib.pyplot as plt
    import torch
    import os

    # 取一个 batch（只用第一个样本）
    images, _ = next(iter(data_loader))
    img = images[0:1].to(device)

    for ep in epochs:
        ckpt_path = os.path.join(output_dir, f'checkpoint_epoch_{ep}.pth')
        if not os.path.exists(ckpt_path):
            print(f"Warning: not found {ckpt_path}")
            continue

        # 加载模型权重
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.to(device).eval()

        # 清空旧 attn_maps
        for m in model.modules():
            if isinstance(m, DSSAWithSSDP):
                m.attn_maps = None

        # 前向一次
        with torch.no_grad():
            _ = model(img)

        # 找到最后一个 DSSAWithSSDP
        dssa_block = next(m for m in model.modules() if isinstance(m, DSSAWithSSDP))
        attn = dssa_block.attn_maps  # [T, 1, heads, N, N]
        attn = attn.mean(dim=2).squeeze(1)  # [T, N, N]

        # 设置一个更大的 figure
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 8"x8" 矢量尺寸
        for t in range(attn.shape[0]):
            ax = axes[t//2, t%2]
            im = ax.imshow(attn[t], cmap='hot', interpolation='nearest')
            ax.set_title(f'Epoch {ep}, t={t+1}')
            ax.axis('off')

        fig.suptitle(f'Attention Maps at Epoch {ep}')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存为高分辨率 PDF（矢量格式）
        save_path = os.path.join(output_dir, f'attn_ep{ep}.pdf')
        fig.savefig(save_path, format='pdf')
        plt.close(fig)
        print(f"Saved high-res PDF attention snapshot: {save_path}")

def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(
        logger, args.distributed_init_mode
    )

    logger.info(str(args))

    ##################################################
    #                   load data
    ##################################################

    dataset_type = args.dataset
    one_hot = None
    if dataset_type == 'CIFAR10':
        num_classes = 10
        input_size = (3, 32, 32)
    elif dataset_type == 'CIFAR10DVS':
        one_hot = 10
        num_classes = 10
        input_size = (3, 64, 64)
    elif dataset_type == 'DVS128Gesture':
        one_hot = 11
        num_classes = 11
        input_size = (3, 64, 64)
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        input_size = (3, 32, 32)
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        input_size = (3, 224, 224)
    elif dataset_type == 'ImageNet100':
        num_classes = 100
        input_size = (3, 224, 224)
    else:
        raise ValueError(dataset_type)
    if len(args.input_size) != 0:
        input_size = args.input_size

    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path,
        args.batch_size,
        args.workers,
        num_classes,
        dataset_type,
        input_size,
        distributed,
        args.augment,
        args.mixup,
        args.cutout,
        args.label_smoothing,
        args.T
    )
    logger.info(
        f'dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}'
    )

    ##################################################
    #                     model
    ##################################################

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    ).cuda()

    if args.transfer:
        ckpt = torch.load(args.transfer, map_location='cpu')
        model.transfer(ckpt['model'])

    ##################################################
    #                   optimizer
    ##################################################

    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ##################################################
    #                  loss functions
    ##################################################

    if args.mixup:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)

    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = CriterionWarpper(criterion_eval)

    ##################################################
    #                     amp
    ##################################################

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    ##################################################
    #               lr scheduler
    ##################################################

    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=1e-5,
        warmup_lr=1e-5,
        warmup_epochs=3,
    )

    ##################################################
    #                   Sync BN
    ##################################################

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ##################################################
    #                     DDP
    ##################################################

    model_without_ddp = model
    if distributed and not args.test_only:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        model_without_ddp = model.module

    ##################################################
    #               custom schedulers
    ##################################################

    scheduler_per_iter = None
    scheduler_per_epoch = None

    ##################################################
    #                     resume
    ##################################################

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        max_acc1 = ckpt['max_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        logger.info(f'Resume from epoch {start_epoch}')
        start_epoch += 1
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################
    if args.test_only:
        if distributed:
            logger.error('Using distributed mode in test, abort')
            return
        test(
            model_without_ddp,
            data_loader_test,
            input_size,
            args,
            logger
        )
        return

    ##################################################
    #        计算需要额外保存的三个关键 epoch
    ##################################################
    # ① warm-up 末期；② 训练中期；③ 训练末期
    warmup_end = model_without_ddp.ssdp.warmup_epochs - 1
    mid_epoch = args.epochs // 2
    final_ep = args.epochs - 1

    ##################################################
    #                     Train
    ##################################################
    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(
            os.path.join(args.output_dir, 'tensorboard'),
            purge_step=start_epoch
        )

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info(f'Epoch [{epoch}] Start, lr {optimizer.param_groups[0]["lr"]:.6f}')

        # 如果模型有 current_epoch，就更新它
        if hasattr(model_without_ddp, 'current_epoch'):
            model_without_ddp.current_epoch = epoch

        # —— 训练一步 ——
        with Timer(' Train', logger):
            train_loss, train_acc1, train_acc5, per_class_acc = train_one_epoch(
                model,
                criterion,
                optimizer,
                data_loader_train,
                logger,
                args.print_freq,
                world_size,
                scheduler_per_iter,
                scaler,
                one_hot
            )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        # —— 验证一步 ——
        with Timer(' Test', logger):
            test_loss, test_acc1, test_acc5 = evaluate(
                model,
                criterion_eval,
                data_loader_test,
                args.print_freq,
                logger,
                one_hot
            )

        # —— TensorBoard 记录 ——
        if is_main_process() and tb_writer is not None:
            tb_record(
                tb_writer,
                train_loss, train_acc1, train_acc5,
                test_loss, test_acc1, test_acc5,
                epoch
            )

        logger.info(
            f' Test loss: {test_loss:.5f}, '
            f'Acc@1: {test_acc1:.5f}, Acc@5: {test_acc5:.5f}'
        )

        # —— 构造 checkpoint 字典 ——
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1,
        }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        # —— 保存 latest 和历史最好 ——
        if args.save_latest:
            save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint_latest.pth')
            )
        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint_max_acc1.pth')
            )

        # —— 新增：在三个关键 epoch 保存专属 checkpoint ——
        if epoch in (warmup_end, mid_epoch, final_ep):
            save_on_master(
                checkpoint,
                os.path.join(
                    args.output_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
            )

    logger.info('Training completed.')

    ##################################################
    #      Visualize Attention Snapshots (PDF)
    ##################################################

    if is_main_process():
        # 三个阶段：warm-up 末期 / 训练中期 / 训练末期
        warmup_end = model_without_ddp.ssdp.warmup_epochs - 1
        mid_epoch  = args.epochs // 2
        final_ep   = args.epochs - 1

        visualize_attention_snapshots(
            model_without_ddp,
            data_loader_test,
            device=next(model_without_ddp.parameters()).device,
            output_dir=args.output_dir,
            epochs=[warmup_end, mid_epoch, final_ep],
        )

    ##################################################
    #                   test
    ##################################################

    del model, model_without_ddp

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    )

    try:
        ckpt = torch.load(
            os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'),
            map_location='cpu'
        )
        model.load_state_dict(ckpt['model'])
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    # reload only test data
    del dataset_train, dataset_test, data_loader_train, data_loader_test
    _, _, _, data_loader_test = load_data(
        args.data_path,
        args.batch_size,
        args.workers,
        num_classes,
        dataset_type,
        input_size,
        False,
        args.augment,
        args.mixup,
        args.cutout,
        args.label_smoothing,
        args.T
    )

    model = model.cuda()
    if is_main_process():
        test(
            model,
            data_loader_test,
            input_size,
            args,
            logger
        )
    logger.info('All Done.')


if __name__ == "__main__":
    main()
