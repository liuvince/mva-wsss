import torch
import numpy as np

# ADAPTED FROM https://github.com/NVlabs/SCOPS
def get_variance(part_map, x_c, y_c):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y


def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor


def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center)
        y_c_value = float(y_center)
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center


def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)


def batch_get_centers(pred_softmax):
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)

def concentration_loss(outputs, outputs_cam):

    B, C, H, W = outputs_cam.shape

    loss = 0
    epsilon = 1e-3
    centers_all = batch_get_centers(outputs_cam)
    for b in range(B):
        centers = centers_all[b]

        for c in torch.nonzero(outputs[b] > 0.5):
            # normalize part map as spatial pdf
            part_map = outputs_cam[b, c[0], :, :] + epsilon  # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map / k
            x_c, y_c = centers[c[0]]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
    return loss / B

# Mine
def entropy_regularization_loss(x):
    return -(torch.sigmoid(x) * torch.log(torch.sigmoid(x) + 1e-5) + (1-torch.sigmoid(x))  * torch.log((1-torch.sigmoid(x)) + 1e-5)).mean()

# https://github.com/facebookresearch/mixup-cifar10
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data_kd(x, x_kd, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    x_kd_a, x_kd_b = x_kd, x_kd[index]
    return mixed_x, x_kd_a, x_kd_b, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# From https://github.com/jiwoon-ahn/psa
class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


import logging
import math

from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
