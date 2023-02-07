import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import math

"""
Learning rate adjustment used for CondenseNet model training
"""
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class MultipleOptimizers(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class MultipleSchedulers(object):
    def __init__(self, *sch):
        self.schedulers = sch

    # def zero_grad(self):
    #     for op in self.optimizers:
    #         op.zero_grad()

    def step(self,value):
        for sch in self.schedulers:
            sch.step(value)

