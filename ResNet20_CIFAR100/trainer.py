import os
import time
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets.folder import make_dataset
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tensorboardX import SummaryWriter
from sbp import SBP


class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.best_prec1 = 0
        self.model = model

        # Training parameters
        new_params = []
        fc_params = []
        paras = dict(self.model.named_parameters())
        for k, v in paras.items():
            if 'fc.weight' in k:
                new_params += [{'params': [v], 'name': [k]}]
            elif 'fc.bias' in k:
                fc_params += [{'params': [v], 'name': [k]}]

        # Define optimizer
        self.optimizer = SBP(new_params, lr=self.args.lr)
        self.bound = [self.model.state_dict()["module.fc.weight_pos"].max(),
                      self.model.state_dict()["module.fc.weight_neg"].min()]

        # define data augment
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # define training set and test set
        self.train_loader = []
        self.test_loader = []
        train_dataset_old = datasets.ImageFolder('./cifar100_80-20/old/train',
                                                 transform=transform_train)
        test_dataset_old = datasets.ImageFolder('./cifar100_80-20/old/test',
                                                transform=transform_test)
        train_dataset_new = datasets.ImageFolder('./cifar100_80-20/new/train',
                                                 transform=transform_train)
        test_dataset_new = datasets.ImageFolder('./cifar100_80-20/new/test',
                                                transform=transform_test)

        train_dataset_new = modify_class(train_dataset_new, 80)
        test_dataset_new = modify_class(test_dataset_new, 80)
        self.train_loader_old = DataLoader(train_dataset_old, batch_size=self.args.bs, shuffle=True)
        self.test_loader_old = DataLoader(test_dataset_old, batch_size=512, shuffle=False)
        self.train_loader_new = DataLoader(train_dataset_new, batch_size=self.args.bs, shuffle=True)
        self.test_loader_new = DataLoader(test_dataset_new, batch_size=512, shuffle=False)

        # dataset for increment learning
        _, old_train_set = random_split(dataset=train_dataset_old, lengths=[31500, 8500])
        _, new_train_set = random_split(dataset=train_dataset_new, lengths=[0, 10000])
        _, old_test_set = random_split(dataset=test_dataset_old, lengths=[0, 8000])
        _, new_test_set = random_split(dataset=test_dataset_new, lengths=[0, 2000])

        ft_train_dataset = ConcatDataset([old_train_set, new_train_set])
        self.ft_train_loader = DataLoader(ft_train_dataset, batch_size=self.args.bs, shuffle=True)
        ft_test_dataset = ConcatDataset([old_test_set, new_test_set])
        self.ft_test_loader = DataLoader(ft_test_dataset, batch_size=512, shuffle=False)

        # Define loss
        self.criterion = nn.CrossEntropyLoss().cuda()

        # Init tensorboard recoder
        self.writer = SummaryWriter('./runs', flush_secs=1)

    def run(self):
        self.model.cuda()

        t_p = 0.2
        t_n = 0.4

        print('Start online training...')
        for epoch in range(self.args.epochs):

            if epoch in [2]:
                t_p += 0.1
                t_n += 0.1
            prec1_train, prec5_train = self.train(self.ft_train_loader, self.model, self.criterion, self.optimizer,
                                                  epoch, t_p=t_p, t_n=t_n)
            self.writer.add_scalar('train_top1_new', prec1_train, epoch)
            self.writer.add_scalar('train_top5_new', prec5_train, epoch)

            # evaluate on validation set
            prec1_test, prec5_test = self.test(self.test_loader_old, self.model, self.criterion, current_data='old')
            self.writer.add_scalar('test_top1_old', prec1_test, epoch)
            self.writer.add_scalar('test_top5_old', prec5_test, epoch)
            prec1_test, prec5_test = self.test(self.test_loader_new, self.model, self.criterion, current_data='new')
            self.writer.add_scalar('test_top1_new', prec1_test, epoch)
            self.writer.add_scalar('test_top5_new', prec5_test, epoch)
            prec1_test, prec5_test = self.test(self.ft_test_loader, self.model, self.criterion, current_data='overall')
            self.writer.add_scalar('test_top1_all', prec1_test, epoch)
            self.writer.add_scalar('test_top5_all', prec5_test, epoch)

            ckp_name = 'model-insitu-incremental-resnet18.th'
            self.save_checkpoint({
                'state_dict': self.model.state_dict()
            }, filename=os.path.join(self.args.save_dir, ckp_name))

    def train(self, train_loader, model, criterion, optimizer, epoch, t_p=0.0, t_n=0.0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1_t = AverageMeter()
        top5_t = AverageMeter()

        model.train()

        end = time.time()

        for batch_id, (data, target) in enumerate(train_loader):

            data_time.update(time.time() - end)

            target = target.cuda()
            data = data.cuda()

            output, x = model(data)

            output_relu = torch.where(output <= 0, torch.zeros_like(output), output)
            one_hot_target = torch.zeros_like(output).cuda().scatter_(1, torch.unsqueeze(target,1), 1) * output.max()
            error = (one_hot_target - output_relu)
            error = torch.where((0 < error) & (error < error.max() * t_p),
                                     torch.zeros_like(error), error)
            error = torch.where((0 > error) & (error > error.min() * t_n),
                                     torch.zeros_like(error), error)

            error = torch.where(error == 0, torch.zeros_like(error), error/error.abs())
            x = torch.where(x.abs() <= x.max().abs() * 0.4, torch.zeros_like(x), x)
            E_update = torch.matmul(error.t(), x)

            loss = criterion(output, target)

            optimizer.zero_grad()

            with torch.no_grad():
                optimizer.step(loss=E_update,
                               weight_pos=self.model.state_dict()["module.fc.weight_pos"],
                               weight_neg=self.model.state_dict()["module.fc.weight_neg"],
                               bound=self.bound,
                               epoch=epoch)
                torch.cuda.empty_cache()

            output = output.float()
            loss = loss.float()

            prec, pred = accuracy(output.data, target, topk=(1,5))

            prec1_t = prec[0]
            prec5_t = prec[1]
            losses.update(loss.item(), data.size(0))
            top1_t.update(prec1_t.item(), data.size(0))
            top5_t.update(prec5_t.item(), data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_id % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(epoch, batch_id, len(train_loader),
                              batch_time=batch_time, data_time=data_time,
                              loss=losses, top1=top1_t, top5=top5_t))

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1_t, top5=top5_t))

        return top1_t.avg, top5_t.avg

    def test(self, test_loader, model, criterion, current_data='old'):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        pred_all = torch.Tensor().cuda().long()
        target_all = torch.Tensor().cuda().long()
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(test_loader):
                target = target.cuda()
                input_var = data.cuda()
                target_var = target.cuda()

                if self.args.half:
                    input_var = input_var.half()

                output, x = model(input_var)
                _, pred = output.data.topk(10, 1, True, True)

                loss = criterion(output, target_var)

                output = output.float()
                loss = loss.float()

                prec, pred = accuracy(output.data, target, topk=(1,5))
                prec1 = prec[0]
                prec5 = prec[1]
                pred_all = torch.cat([pred_all, pred[0]])
                target_all = torch.cat([target_all, target])
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_id % self.args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                          .format(batch_id, len(test_loader),
                                  batch_time=batch_time, loss=losses,
                                  top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}, CurrentTask: {current_data}'
                  .format(top1=top1, top5=top5, current_data=current_data))

        return top1.avg, top5.avg

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def modify_class(dataset, idx):
    for i in dataset.classes:
        dataset.class_to_idx[i] += idx
    dataset.samples = make_dataset(dataset.root,
                                   dataset.class_to_idx,
                                   dataset.extensions,
                                   None)
    dataset.targets = [s[1] for s in dataset.samples]
    return dataset
