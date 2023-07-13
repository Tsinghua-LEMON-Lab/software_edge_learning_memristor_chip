import math
import torch
from torch.optim.optimizer import Optimizer
from models.utilities.common import get_pulse


class SBP(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.current_set = True
        super(SBP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SBP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, loss=None, weight_pos=None, weight_neg=None, bound=None, epoch=None):
        for group in self.param_groups:
            for p in group['params']:

                update = loss
                if update.size() == torch.Size([100, 512]):
                    frozen_weight_matrix = torch.cat([torch.zeros(80, 512), torch.ones(20, 512)]).cuda()
                    update *= frozen_weight_matrix
                p.data = self.weight_update(update, weight_pos, weight_neg, bound, epoch)

    def weight_update(self, update, weight_pos, weight_neg, bound, epoch):
        update = torch.round(update)
        pulse_nums = torch.abs(update)
        if epoch < 2:
            pulse_nums = torch.div(pulse_nums, 1, rounding_mode='floor')
        elif epoch < 7:
            pulse_nums = torch.div(pulse_nums, 2, rounding_mode='floor')
        else:
            pulse_nums = torch.div(pulse_nums, 4, rounding_mode='floor')
        if self.current_set:
            pulse_pos_set = get_pulse(weight_pos, 'set')
            pulse_neg_set = get_pulse(weight_neg, 'set')
            weight_pos += torch.where(update > 0, pulse_pos_set*pulse_nums, torch.zeros_like(update))
            weight_neg -= torch.where(update < 0, pulse_neg_set*pulse_nums, torch.zeros_like(update))
        else:
            pulse_pos_reset = get_pulse(weight_pos, 'reset')
            pulse_neg_reset = get_pulse(weight_neg, 'reset')
            weight_pos -= torch.where(update < 0, pulse_pos_reset*pulse_nums, torch.zeros_like(update))
            weight_neg += torch.where(update > 0, pulse_neg_reset*pulse_nums, torch.zeros_like(update))

        self.change_current_opt()

        return torch.clamp(weight_pos, max=bound[0], min=0) + torch.clamp(weight_neg, min=bound[1], max=0)

    def change_current_opt(self):
        self.current_set = not self.current_set
