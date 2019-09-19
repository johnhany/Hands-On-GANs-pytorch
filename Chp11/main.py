import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from PIL import Image

import utils

from build_gan import Model
from datasets import ShapeNetDataset

FLAGS = None

def main():
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")

    print('Loading data...\n')
    dataloader = DataLoader(ShapeNetDataset(FLAGS.data_dir,
                                            FLAGS.cube_len),
                            batch_size=FLAGS.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    print('Creating model...\n')
    model = Model(FLAGS.model, device, dataloader, FLAGS.latent_dim, FLAGS.cube_len)
    model.create_optim(FLAGS.g_lr, FLAGS.d_lr)
    # print(model.generator)
    # print(model.discriminator)

    # Train
    model.train(FLAGS.epochs, FLAGS.d_loss_thresh, FLAGS.log_interval, FLAGS.export_interval, FLAGS.out_dir, True)

    model.save_to('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hands-On GANs - Chapter 11')
    parser.add_argument('--model', type=str, default='3dgan', help='3dgan')
    parser.add_argument('--cuda', type=utils.boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=utils.boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str, default='/media/john/DataAsgard/3d_models/volumetric_data/chair/30/train', help='Directory for dataset.')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batches in training')
    parser.add_argument('--g_lr', type=float, default=0.0025, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for discriminator')
    parser.add_argument('--d_loss_thresh', type=float, default=0.8, help='threshold for updating D')
    parser.add_argument('--cube_len', type=int, default=32, help='size of cube')
    parser.add_argument('--latent_dim', type=int, default=200, help='length of latent vector')
    parser.add_argument('--log_interval', type=int, default=100, help='iteration interval between logging')
    parser.add_argument('--export_interval', type=int, default=10, help='epoch interval between exporting images and trained models')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        utils.clear_folder(FLAGS.out_dir)

    log_file = os.path.join(FLAGS.out_dir, 'log.txt')
    print("Logging to {}\n".format(log_file))
    sys.stdout = utils.StdOut(log_file)

    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))

    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    main()
