import argparse

parser = argparse.ArgumentParser(description='Propert MLP for MNIST in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=7, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--test_only', dest='test_only', action='store_true',
                    help='test only model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pre_train', type=str, default='./saved/resnet18-5c106cde.pth',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./saved', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=2)
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--weight_k_bits', type=int, default=4,
                    help='The k_bits of the weight to be quantized')
parser.add_argument('--feature_k_bits', type=int, default=8,
                    help='The k_bits of the feature map to be quantized')
parser.add_argument('--noise', type=bool, default=True, help='apply noise')
parser.add_argument('--noise_type', type=str, default='ratio', choices=('ratio', 'add'), help='noise type')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')



# global args, best_prec1

args = parser.parse_args()
