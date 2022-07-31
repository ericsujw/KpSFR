"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os


def convert_arg_line_to_args(arg_line):
    line_all = arg_line.split()
    for idx, arg in enumerate(line_all):
        if len(line_all) > 1 and line_all[1] == 'False':
            return
        # ignore only resume_weight
        if len(line_all) == 1:
            return
        if arg[0] == '[' or arg == 'False' or arg == 'True':
            continue
        if arg == '--isTrain':
            continue
        if not arg.strip():
            continue

        yield arg


class CustomOptions():

    def __init__(self, train):
        self.initialized = False
        self.isTrain = train

    def initialize(self, parser):
        # Experiment specifics
        parser.add_argument('--checkpoints_dir', type=str,
                            default='checkpoint')
        parser.add_argument('--name', type=str, default='debug/check_full_model',
                            help='name of the experiment. It decides where to store samples and models')

        # Data related
        parser.add_argument('--public_worldcup_root', type=str,
                            default='./dataset/soccer_worldcup_2014/soccer_data', help='data root of public worldcup dataset')
        parser.add_argument('--custom_worldcup_root', type=str,
                            default='./dataset/WorldCup_2014_2018', help='data root of custom worldcup dataset')
        parser.add_argument('--template_path', type=str,
                            default='./assets', help='path of worldcup template')
        parser.add_argument('--trainset', type=str,
                            default='train_val', help='path of training set')
        parser.add_argument('--testset', type=str,
                            default='test', help='path of testing set')

        # Training related
        parser.add_argument('--gpu_ids', type=str, default='1',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--model_archi', type=str,
                            default='KC_STCN', help='KC_STCN or KC or STCN')
        parser.add_argument('--loss_mode', type=str,
                            default='all', help='all or dice_bce or dice_wce')
        parser.add_argument('--use_non_local', type=int,
                            default=1, help='if use non local block layer')
        parser.add_argument('--train_epochs', type=int, default=300)
        parser.add_argument('--train_stage', type=int, default=0,
                            help='training stage (0-public, 1-custom)')
        parser.add_argument('--num_objects', type=int,
                            default=4, help='the number of objects to be segmented')
        parser.add_argument('--resume', action='store_true',
                            default=False, help='if resume training')
        parser.add_argument('--ckpt_path', type=str,
                            default='', help='path of pretrained or resumed weight')
        parser.add_argument('--sfp_finetuned', action='store_true',
                            default=False, help='if use finetuned results of single frame prediction on testing')

        # Hyperparameters
        parser.add_argument('--batch_size', type=int,
                            default=4, help='input batch size')
        parser.add_argument('--train_lr', type=float,
                            default=1e-4, help='base learning rate')
        parser.add_argument('--step_size', type=int,
                            default=200, help='learning rate scheduling')
        parser.add_argument('--weight_decay', type=float,
                            default=0.0, help='if the need for regularization')
        parser.add_argument('--nms_thres', type=float,
                            default=0.25, help='threshold when calculating nms')  # 0.995
        parser.add_argument('--pr_thres', type=float,
                            default=5.0, help='threshold when calculating precision and recall of keypoints')
        parser.add_argument('--noise_trans', type=float,
                            default=5.0, help='noise parameter translation')
        parser.add_argument('--noise_rotate', type=float,
                            default=0.014, help='noise parameter rotation')  # 0.0084

        # Inference settings
        parser.add_argument('--target_video', type=str, nargs='+',
                            default=[], help='Inference single video only, default is None')
        parser.add_argument('--target_image', type=str, nargs='+',
                            default=[], help='Inference single image only, default is None')

        self.initialized = True
        return parser

    def gather_options(self):

        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
            parser.convert_arg_line_to_args = convert_arg_line_to_args
            parser = self.initialize(parser)

        if sys.argv.__len__() == 2:
            arg_filename_with_prefix = '@' + sys.argv[1]
            opt, unknown = parser.parse_known_args([arg_filename_with_prefix])
            opt = parser.parse_args([arg_filename_with_prefix])
        else:
            # get the basic options
            opt, unknown = parser.parse_known_args()
            opt = parser.parse_args()

        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:<25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt')

        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default:%s]' % str(default)
                opt_file.write(
                    '--{:<25} {:<30} {}\n'.format(str(k), str(v), comment))

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # os._exit(0)
        self.opt = opt
        return self.opt
