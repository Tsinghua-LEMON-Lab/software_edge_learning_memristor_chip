import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SpotDataset(object):
    def __init__(self, path_root, args):
        self.root_path = path_root
        self.target_h = args.h
        self.target_w = args.w
        self.ROI_NN = [45, 0, 150, 400]
        self.data_index = pd.read_csv(self.root_path).values

    def preprecessImg(self, img):
        img = img[int(self.ROI_NN[0]):int(self.ROI_NN[0] + self.ROI_NN[2]),
              int(self.ROI_NN[1]):int(self.ROI_NN[1] + self.ROI_NN[3])]
        img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        img = img / 255.
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, index):
        img_path, angle, throttle = self.data_index[index]
        img = cv2.imread(img_path)
        img = self.preprecessImg(img)
        angle = torch.FloatTensor([(angle + 1) / 2]).expand(8)
        throttle = torch.LongTensor([throttle])
        return img, angle, throttle

    def __len__(self):
        return len(self.data_index)


class Binary_a(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = (torch.sign(input) + 1) / 2
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class Ternary(Function):

    @staticmethod
    def forward(self, input):
        E = torch.abs(input).mean()
        threshold = E * 0.7
        output = torch.sign(
            torch.add(torch.sign(torch.add(input, threshold)), torch.sign(torch.add(input, -threshold))))
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        grad_input = grad_output.clone()
        return grad_input


class activation_bin(nn.Module):
    def __init__(self, A, args):
        super().__init__()
        self.A = A
        self.args = args

    def binary(self, input):
        output = Binary_a.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
        else:
            output = input
        return output


class weight_tnn_bin(nn.Module):
    def __init__(self, W=2, args=None):
        super().__init__()
        self.W = W
        self.args = args

    def ternary(self, input):
        output = Ternary.apply(input)
        return output

    def forward(self, input):
        if self.W == 3:
            output_fp = input.clone()
            output, threshold = self.ternary(input)
            output_abs = torch.abs(output_fp)
            mask_le = output_abs.le(threshold)
            mask_gt = output_abs.gt(threshold)
            output_abs[mask_le] = 0
            output_abs_th = output_abs.clone()
            output_abs_th_sum = output_abs_th.sum().float()
            mask_gt_sum = mask_gt.sum().float()
            alpha = output_abs_th_sum / mask_gt_sum
            if self.args.noise:
                output = output + self.args.noise_sigma * torch.randn_like(output) / 0.4
            output = output * alpha
        else:
            output = input
        return output


class my_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias, A=2, W=2, args=None):
        super(my_Linear, self).__init__(in_features, out_features, bias=bias)
        self.A = A
        self.args = args
        self.activation_quantizer = activation_bin(A, self.args)
        self.weight_quantizer = weight_tnn_bin(W, self.args)

    def forward(self, input):
        bin_input = self.activation_quantizer(input)
        tnn_bin_weight = self.weight_quantizer(self.weight)
        output = F.linear(bin_input, tnn_bin_weight, self.bias)
        if self.A == 2:
            return output, bin_input
        else:
            return output


class MultiTaskModel(nn.Module):

    def __init__(self, args):
        super(MultiTaskModel, self).__init__()
        args.outputs_txt = []
        args.save_flag = True
        self.args = args
        self.args = args
        self.quan = args.quan
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        )

        self.sigmoid = nn.Sigmoid()

        self.fc1 = my_Linear(32 * 4 * 4, 100, bias=False, A=2, W=3, args=self.args) if self.quan else nn.Linear(
            32 * 4 * 4, 100, bias=False)
        self.fc2_1 = my_Linear(100, 8, bias=False, A=32, W=3, args=self.args) if self.quan else nn.Linear(100, 8,
                                                                                                          bias=False)
        self.fc2_2 = my_Linear(100, 2, bias=False, A=32, W=3, args=self.args) if self.quan else nn.Linear(100, 2,
                                                                                                          bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, stage=None):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        conv_outs = self.fc1(x)
        if stage:
            return conv_outs[1]
        if self.args.quan:
            x = F.relu(conv_outs[0])
        else:
            x = F.relu(conv_outs)
        angle = F.relu(self.fc2_1(x))
        throttle = F.relu(self.fc2_2(x))
        return angle, throttle


def setTestArguments():
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('--path_root', default='datasets', type=str, help='folder of dataset')
    parser.add_argument('--load_which_model', default='trained_model.pth', type=str, help='number of epochs to train')
    parser.add_argument('--test_batch_size', default=32, type=int, help='batch size for testing (default:200)')
    parser.add_argument('--w', default=224, type=int, help='')
    parser.add_argument('--h', default=224, type=int, help='')
    parser.add_argument('--save_model', default='checkpoints', type=str, help='')
    parser.add_argument('--quan', default=True, type=bool, help='whether to quantification')
    parser.add_argument('--noise', default=True, type=bool, help='Whether to add noise in weight')
    parser.add_argument('--noise_sigma', default=0.05, type=float, help='Noise amplitude')

    args = parser.parse_args()
    if args.quan:
        args.save_model = os.path.join(args.save_model, 'quant')
    os.makedirs(args.save_model, exist_ok=True)
    return args


def calTestAcc(args):
    dataset_test_off = SpotDataset(os.path.join(args.path_root, 'dark_scene_test_index.csv'), args)
    dataloader_test_off = DataLoader(dataset_test_off, batch_size=args.test_batch_size)
    dataset_test_on = SpotDataset(os.path.join(args.path_root, 'bright_scene_test_index.csv'), args)
    dataloader_test_on = DataLoader(dataset_test_on, batch_size=args.test_batch_size)

    model = MultiTaskModel(args).to(args.device)
    model.load_state_dict(
        torch.load(os.path.join(args.save_model, args.load_which_model), map_location=torch.device('cpu')))
    model.eval()

    accuracy_count = 0
    with torch.no_grad():
        for inputs, _, targets_throttle in tqdm(dataloader_test_off, total=len(dataloader_test_off)):
            inputs, targets_throttle = inputs.to(device), targets_throttle.to(device)
            predictions_angle, predictions_throttle = model(inputs)
            predictions = torch.max(predictions_throttle.cpu(), 1)[1]
            targets = targets_throttle.cpu().reshape(-1)
            accuracy_count += (predictions == targets).sum()

    accuracy_quant_off = accuracy_count * 1. / len(dataloader_test_off.dataset)

    accuracy_count = 0
    with torch.no_grad():
        for inputs, _, targets_throttle in tqdm(dataloader_test_on, total=len(dataloader_test_on)):
            inputs, targets_throttle = inputs.to(device), targets_throttle.to(device)
            predictions_angle, predictions_throttle = model(inputs)
            predictions = torch.max(predictions_throttle.cpu(), 1)[1]
            targets = targets_throttle.cpu().reshape(-1)
            accuracy_count += (predictions == targets).sum()

    accuracy_quant_on = accuracy_count * 1. / len(dataloader_test_on.dataset)

    print('test dark scene throttles accuracy : %.4f' % (accuracy_quant_off))
    print('test bright scene throttles accuracy : %.4f' % (accuracy_quant_on))


if __name__ == '__main__':
    args = setTestArguments()
    args.device = device
    calTestAcc(args)
