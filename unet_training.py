import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms




def Sobel_edge_loss(inputs, target):

    if target.dim() == 3:
        target = target.unsqueeze(1)

    inputs = torch.softmax(inputs, dim=1)[:, 1:2, :, :]

    sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]]], dtype=torch.float32, device=inputs.device)

    sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]]]], dtype=torch.float32, device=inputs.device)

    # 提取边缘
    pred_grad_x = F.conv2d(inputs, sobel_kernel_x, padding=1)
    pred_grad_y = F.conv2d(inputs, sobel_kernel_y, padding=1)
    pred_edge = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)

    # 提取边缘
    target_grad_x = F.conv2d(target.float(), sobel_kernel_x, padding=1)
    target_grad_y = F.conv2d(target.float(), sobel_kernel_y, padding=1)
    target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)

    edge_loss = F.mse_loss(pred_edge, target_edge)

    return edge_loss




def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
