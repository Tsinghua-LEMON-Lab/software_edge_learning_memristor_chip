import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class ResBlock(nn.Module):
    def __init__(self, conv, in_feats, out_feats, kernel_size, bias=False, is_res=False, act=nn.ReLU(True), stride=1):

        super(ResBlock, self).__init__()

        self.is_res = is_res
        m = []

        m.append(conv(in_feats, out_feats, kernel_size, bias=bias, stride=stride))
        m.append(nn.BatchNorm2d(out_feats))
        m.append(act)
        m.append(conv(out_feats, out_feats, kernel_size, bias=bias))
        m.append(nn.BatchNorm2d(out_feats))

        res = []
        if self.is_res:
            res.append(conv(in_feats, out_feats, 1, stride=stride, bias=bias))
            res.append(nn.BatchNorm2d(out_feats))
            self.res = nn.Sequential(*res)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        if self.is_res:
            res += self.res(x)
        else:
            res += x
        res = F.relu(res)
        return res


def mapping_2t2r(weight):
    min_w = (weight.max() - weight.min()) / 36 * 2
    weight_pos = nn.Parameter(torch.ones_like(weight))
    weight_neg = nn.Parameter(torch.ones_like(weight))
    weight_pos.data = torch.where(weight >= 0, weight + min_w, min_w)
    weight_neg.data = torch.where(weight < 0, weight - min_w, -1*min_w)
    return weight_pos, weight_neg


def get_pulse(weight, opt):
    reset_mean_G_matrix = torch.Tensor([-5.10919102040816e-07, -2.69156862857143e-07, -1.45545705494506e-07,
                                        -9.8736975e-08, -5.4099215037594e-08, -4.5403761038961e-08,
                                        -2.18754331428571e-08, -7.44087038095238e-08, -7.64108873469388e-08,
                                        -2.75467392857143e-09]).cuda()
    set_mean_G_matrix = torch.Tensor([3.25106657142857e-07,2.49759040000000e-07,2.36664794285714e-07,
                                    1.43037453333333e-07,8.66050671428572e-08,5.98222434285714e-08,
                                    5.38902411428571e-08,3.58482819047619e-08,1.85277600000000e-08,
                                    1.94062107142857e-08]).cuda()
    reset_std_G_matrix = torch.Tensor([9.05607477777778e-07, 7.95225566666667e-07, 6.815025e-07,
                                       6.47670066666667e-07, 5.61997366666667e-07, 5.91771366666667e-07,
                                       4.99325133333333e-07, 4.42767911111111e-07, 5.18210288888889e-07,
                                       3.79624877777778e-07]).cuda()
    set_std_G_matrix = torch.Tensor([5.26255757575758e-07,5.39530269360269e-07,5.80772070707071e-07,
                                   6.01795134680135e-07,6.49712222222222e-07,6.39751616161616e-07,
                                   6.65645134680135e-07,6.35491734006734e-07,6.28778973063973e-07,
                                   4.40424074074074e-07]).cuda()

    index = ((weight - weight.min()) / (weight.max() - weight.min()) * 10).long()
    # torch.cuda.empty_cache()
    index = torch.where(index == 10, index - 1, index)
    if opt == 'set':
        set_mean_matrix = set_mean_G_matrix / 18 * (weight.max() - weight.min()) * 1e6
        set_std_matrix = set_std_G_matrix / 18 * (weight.max() - weight.min()) * 1e6 / torch.sqrt(
            torch.Tensor([2.])).cuda()
        set_mean = set_mean_matrix[index]
        set_std = set_std_matrix[index]
        return torch.clamp(torch.normal(torch.ones_like(set_mean)), min=-1., max=1.) * set_std + set_mean
    elif opt == 'reset':
        reset_mean_matrix = reset_mean_G_matrix / 18 * (weight.max() - weight.min()) * 1e6
        reset_std_matrix = reset_std_G_matrix / 18 * (weight.max() - weight.min()) * 1e6 / torch.sqrt(
            torch.Tensor([2.])).cuda()
        reset_mean = reset_mean_matrix[index]
        reset_std = reset_std_matrix[index]
        return torch.clamp(torch.normal(torch.ones_like(reset_mean)), min=-1., max=1.) * reset_std + reset_mean
    else:
        raise AttributeError("Wrong operation!")
