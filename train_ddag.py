# Copyright 2021 @Zhangzhiwei
# Code import from https://github.com/mangye16/DDAG

import os
import time
import argparse
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.dataset.transforms.py_transforms import Compose

from utils.utils import *
from PIL import Image
from data.data_loader import SYSUDatasetGenerator
from data.data_manager import *
from data.data_loader import *
from model.model_main import *


def print_dataset_info(dataset_type, trainset,query_label, gall_label, start_time):
    n_class = len(np.unique(trainset.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)
    print('Dataset {} statistics:'.format(dataset_type))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - start_time))

def train():
    pass

def test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDAG Code Mindspore Version")
    parser.add_argument("--dataset", default='sysu', help='dataset name: regdb or sysu')
    parser.add_argument("--lr", default=0.1, type=float, help='learning rate, 0.00035 for adam')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained resnet-50 checkpoint path')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--model_path', default='save_model/', type=str,
                        help='model save path')
    parser.add_argument('--debug', action="store_true", 
                        help='if set true, use a demo dataset for debugging')
    parser.add_argument('--save_epoch', default=20, type=int,
                        metavar='s', help='save model every 10 epochs')
    parser.add_argument('--log_path', default='log/', type=str,
                        help='log save path')
    parser.add_argument('--vis_log_path', default='log/vis_log_ddag/', type=str,
                        help='log save path')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--low-dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--img_w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                        metavar='imgh', help='img height')
    parser.add_argument('--batch-size', default=8, type=int,
                        metavar='B', help='training batch size')
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--part', default=3, type=int,
                        metavar='tb', help=' part number')
    parser.add_argument('--method', default='id+tri', type=str,
                        metavar='m', help='method type')
    parser.add_argument('--drop', default=0.2, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--margin', default=0.3, type=float,
                        metavar='margin', help='triplet loss margin')
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--seed', default=0, type=int,
                        metavar='t', help='random seed')
    parser.add_argument('--gpu', default='', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--lambda0', default=1.0, type=float,
                        metavar='lambda0', help='graph attention weights')
    parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
    parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')

    args = parser.parse_args()

    # check whether gpu is available 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.gpu) > 0:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        if config.run_distribute:
            init("nccl")
            # context.set_auto_parallel_context(device_num=get_group_size(),
            #                                   parallel_mode=ParallelMode.DATA_PARALLEL,
            #                                   gradients_mean=True)
            context.set_auto_parallel_context(device_num=args.gpu,
                                            parallel_mode=ParallelMode.DATA_PARALLEL,
                                            gradients_mean=True)
    # Parallel Mode can set Hybrid_Parallel Mode
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    

    dataset_type = args.dataset

    if dataset_type == "sysu":
        # TODO: define your data path
        # don't forget add '/' at the end of folder path
        data_path = './data/SYSU-MM01/'
        # log_path
        # test_mode = [1,2] # infrared tp visible
    elif dataset_type == "regdb":
        # TODO: define your data path
        data_path = "./data/REGDB"
        # log_path
        # test_mode = [2,1] # visible tp infared

    checkpoint_path = args.model_path

    log_path = args.log_path

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.isdir(args.vis_log_path):
        os.makedirs(args.vis_log_path)

    # log file name
    suffix = dataset_type
    if args.graph:
        suffix = suffix + '_G'
    if args.wpa:
        suffix = suffix + '_P_{}'.format(args.part)
    suffix = suffix + '_drop_{}_{}_{}_lr_{}_seed_{}'.format(args.drop, args.num_pos, args.batch_size, args.lr, args.seed)
    if not args.optim == 'sgd':
        suffix = suffix + '_' + args.optim
    if dataset_type == 'regdb':
        suffix = suffix + '_trial_{}'.format(args.trial)

    # test_log_file = open(log_path + suffix + '.txt', "w")
    # sys.stdout = Logger(log_path + suffix + '_os.txt')

    # vis_log_dir = args.vis_log_path    + suffix + '/'
    # if not os.path.isdir(vis_log_dir):
    #     os.makedirs(vis_log_dir)
    # writer = SummaryWriter(vis_log_dir)

    
    best_acc = 0
    best_acc = 0  # best test accuracy
    start_epoch = 0
    feature_dim = args.low_dim
    wG = 0
    start_time = time.time()

    print("==> Loading data")
    # Data Loading code

    transform_train = Compose(
        [
            py_trans.Decode(),
            py_trans.Pad(10),
            py_trans.RandomCrop((args.img_h, args.img_w)),
            py_trans.RandomHorizontalFlip(),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    ) 

    transform_test = Compose(
        [
            py_trans.Decode(),
            py_trans.Resize((args.img_h, args.img_w)),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    

    if dataset_type=="sysu":
        # train_set
        trainset_generator =  SYSUDatasetGenerator(data_dir=data_path, ifDebug=args.debug)
        color_pos, thermal_pos = GenIdx(trainset_generator.train_color_label, trainset_generator.train_thermal_label)
        
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    gallset = TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, img_size=(args.img_w, args.img_h))

    print_dataset_info(dataset_type, trainset_generator, query_label, gall_label, start_time)
    
    # Model Building
    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)
    net = embed_net(args.low_dim, n_class, drop=args.drop, part=args.part, arch=args.arch, wpa=args.wpa)
    # TODO: voncert to mindspore
    # net.to(device) 
    # cudnn.benchmark = True
    
    # define loss function
    criterion1 = nn.SoftmaxCrossEntropyWithLogits()
    loader_batch = args.batch_size * args.num_pos
    # criterion2 = 

    # optimizer
    if args.optim == 'sgd':
        ignored_params = list(map(id, net.bottleneck.get_parameters())) \
                        + list(map(id, net.classifier.get_parameters())) \
                        + list(map(id, net.wpa.get_parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.get_parameters())

        optimizer_P = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.get_parameters(), 'lr': args.lr},
            {'params': net.classifier.get_parameters(), 'lr': args.lr},
            {'params': net.wpa.get_parameters(), 'lr': args.lr},
            # {'params': net.attention_0.parameters(), 'lr': args.lr},
            # {'params': net.attention_1.parameters(), 'lr': args.lr},
            # {'params': net.attention_2.parameters(), 'lr': args.lr},
            # {'params': net.attention_3.parameters(), 'lr': args.lr},
            # {'params': net.out_att.parameters(), 'lr': args.lr} ,
            ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

    # optimizer_G = optim.SGD([
    #     {'params': net.attention_0.parameters(), 'lr': args.lr},
    #     {'params': net.attention_1.parameters(), 'lr': args.lr},
    #     {'params': net.attention_2.parameters(), 'lr': args.lr},
    #     {'params': net.attention_3.parameters(), 'lr': args.lr},
    #     {'params': net.out_att.parameters(), 'lr': args.lr}, ],
    #     weight_decay=5e-4, momentum=0.9, nesterov=True)    


    # training
    print('==> Start Training...')
    for epoch in range(start_epoch, 81 - start_epoch):

        print('==> Preparing Data Loader...')
        # identity sampler: 
        sampler = IdentitySampler(trainset_generator.train_color_label, \
                                trainset_generator.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                                epoch)
        trainset = ds.GeneratorDataset(trainset_generator, ["data", "label"], sampler=sampler).map(
            operations=transform_train, input_columns=["data"]
        )
        
        # trainset.cIndex = sampler.index1  # color index
        # trainset.tIndex = sampler.index2  # infrared index
        print(epoch)
        print(sampler.cIndex)
        print(sampler.tIndex)

        loader_batch = args.batch_size * args.num_pos

#     trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
#                                   sampler=sampler, num_workers=args.workers, drop_last=True)

#     # training
#     wG = train(epoch, wG)

#     if epoch > 0 and epoch % 2 == 0:
#         print('Test Epoch: {}'.format(epoch))
#         print('Test Epoch: {}'.format(epoch), file=test_log_file)

#         # testing
#         cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
#         # log output
#         print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#         print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP), file=test_log_file)

#         print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
#         print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att), file=test_log_file)
#         test_log_file.flush()
        
#         # save model
#         if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
#             best_acc = cmc_att[0]
#             state = {
#                 'net': net.state_dict(),
#                 'cmc': cmc_att,
#                 'mAP': mAP_att,
#                 'epoch': epoch,
#             }
#             torch.save(state, checkpoint_path + suffix + '_best.t')