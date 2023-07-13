from option import args
from trainer import Trainer
from collections import OrderedDict
import torch
import torch.nn as nn
from models.resnet_quant import resnet18_quant
from models.utilities.common import mapping_2t2r
import os

args.seed = 4096
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)


def add_new_cat(ckp, old_cat):
    for k in ckp:
        if 'fc.weight' in k:
            stdv_w = ckp[k][:old_cat].std()
            mean_w = ckp[k][:old_cat].mean()
            new_para_w = nn.Parameter(torch.Tensor(100 - old_cat, 512)).data.uniform_(-stdv_w, stdv_w).cuda() + mean_w
            ckp[k] = torch.cat([ckp[k][:old_cat], new_para_w])
    return ckp


def main():
    args.cat_num = 100
    args.pre_train_ckp = OrderedDict()
    args.pre_train_old = "./saved/model-80.th"
    ckpt_old = torch.load(args.pre_train_old)['state_dict'] \
        if 'state_dict' in torch.load(args.pre_train_old) else torch.load(args.pre_train_old)

    ckpt = add_new_cat(ckpt_old, 80)
    for k in ckpt:
        args.pre_train_ckp[k] = ckpt[k]
    args.pre_train_ckp["module.fc.weight_pos"], args.pre_train_ckp["module.fc.weight_neg"] \
        = mapping_2t2r(args.pre_train_ckp["module.fc.weight"])
    model = resnet18_quant(args)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(args.pre_train_ckp, strict=True)
    train = Trainer(args, model)
    train.run()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
