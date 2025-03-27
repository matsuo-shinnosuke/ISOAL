import argparse
from pathlib import Path
from utils import set_device

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model_backbone', default='resnet18', type=str, choices=['resnet18', 'resnet50'])
    parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd','adam'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--output_dir', default='result/', type=str)

    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'cub200'])
    parser.add_argument('--uncertainty', default='margin', type=str,  choices=['ent', 'max_conf', 'margin'])
    parser.add_argument('--num_rounds', default=5, type=int)
    parser.add_argument('--budget', default=1000, type=int)
    parser.add_argument('--cost_weak', default=0.5, type=float) # cost_full=1.0

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.device = set_device(args)

    if args.dataset == 'cifar100':
        args.num_classes, args.num_super_classes = 100, 20
        args.image_size = 32
        # args.budget = 1000
    elif args.dataset == 'cub200':
        args.num_classes, args.num_super_classes = 200, 70
        args.image_size = 64
        # args.budget = 500

    return args