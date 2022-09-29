import math
import numpy as np


def adjust_learning_rate(config, optimizer, epoch):

    lr = config.lr
    if config.lr_cosine:
        eta_min = lr * (config.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / config.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(config.lr_decay_epochs))
        if steps > 0:
            lr = lr * (config.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(config, epoch, batch_id, total_batches, optimizer):

    if config.lr_warm and epoch < config.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (config.lr_warm_epochs * total_batches)
        lr = config.lr_warmup_from + p * (config.lr_warmup_to - config.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate
