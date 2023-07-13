import math
import torch
import torch.nn as nn
from models.utilities.common import mapping_2t2r


def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8


def torch_round():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply


class QuantAct(nn.Module):
    """
    Quantization function for quantize activation with maximum.
    """
    def __init__(self, k_bits):
        super(QuantAct, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = torch_round()

    def forward(self, input):
        max_val = quant_max(input)
        act = input * self.qmax / max_val
        q_act = self.round(act)
        q_act = q_act * max_val / self.qmax
        return q_act


class QuantWeight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(QuantWeight, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = torch_round()

    def forward(self, input):
        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = self.round(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight


class QuantConv2d(nn.Module):
    """
    A convolution layer with quantized weight.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, weight_k_bits=32, feature_k_bits=32, noise=False, noise_type=None):
        super(QuantConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.noise = noise
        self.noise_type = noise_type

        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.weight_k_bits = weight_k_bits
        self.feature_k_bits = feature_k_bits
        self.QuantWeight = QuantWeight(k_bits=weight_k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n = n * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, x):
        weight_q = self.weight
        if self.noise:
            noise = torch.randn_like(weight_q).clamp_(-3.0, 3.0)
            if self.noise_type == 'add':
                weight_n = weight_q + (weight_q.max() - weight_q.min()) * noise * 0.1
            else:
                weight_n = weight_q + weight_q * noise * 0.05
            weight_n = (weight_n - weight_q).detach()+weight_q
        else:
            weight_n = weight_q
        res = nn.functional.conv2d(x, weight_n, self.bias, self.stride, self.padding,
                                   self.dilation, self.groups)
        return res


class Linear2T2R(nn.Module):
    """
    A Linear layer with quantized weight.
    """
    def __init__(self, in_channels, out_channels, bias=False, noise=False, noise_type=None):
        super(Linear2T2R, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        self.in_channels = in_channels
        self.noise = noise
        self.noise_type = noise_type
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.output = None
        self.QuantWeight = QuantWeight(k_bits=4)
        self.reset_parameter()
        self.weight_pos, self.weight_neg = mapping_2t2r(self.weight)

    def reset_parameters(self):
        n = self.in_channels
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, x):
        weight_q = self.weight
        if self.noise:
            noise = torch.randn_like(weight_q).clamp_(-3.0, 3.0)
            if self.noise_type == 'add':
                weight_n = weight_q + weight_q.max() * noise * 0.05
            else:
                weight_n = weight_q + weight_q * noise * 0.05
            weight_n = (weight_n - weight_q).detach() + weight_q
        else:
            weight_n = weight_q
        res = nn.functional.linear(x, weight_n, self.bias)
        return res


def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def quant_conv3x3(in_channels, out_channels, kernel_size=3, padding=1, stride=1, weight_k_bits=4,
                  feature_k_bits=8, bias=False, noise=False, noise_type=None):
    return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, feature_k_bits=feature_k_bits, weight_k_bits=weight_k_bits, bias=bias,
                       noise=noise, noise_type=noise_type)


def quant_conv1x1(in_channels, out_channels, kernel_size=1, padding=0, stride=1, weight_k_bits=4,
                  feature_k_bits=8, bias=False, noise=False, noise_type=None):
    return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, feature_k_bits=feature_k_bits, weight_k_bits=weight_k_bits, bias=bias,
                       noise=noise, noise_type=noise_type)


def linear_2t2r(in_channels, out_channels, bias=False, noise=False, noise_type=None):
    return Linear2T2R(in_channels=in_channels, out_channels=out_channels, bias=bias, noise=noise, noise_type=noise_type)