# Copyright 2021 @Zhangzhiwei
# Code import from https://github.com/mangye16/DDAG

import os
import time
import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.vision.py_transforms as py_trans


from mindspore import context, Model, load_checkpoint, load_param_into_net, DatasetHelper, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.train.callback import LossMonitor
from mindspore.nn import Momentum, TrainOneStepCell, WithLossCell

from data.data_loader import SYSUDatasetGenerator
from data.data_manager import *
from data.data_loader import *
from model.model_main import *
from model.trainingCell import MyWithLossCell
from utils.utils import *
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser(description="DDAG Code Mindspore Version")
     # dataset settings
    parser.add_argument("--dataset", default='SYSU', choices=['SYSU', 'RegDB'],
                        help='dataset name: RegDB or SYSU')
    parser.add_argument('--data_path',type=str, default=None )
    # Only used on Huawei Cloud OBS service,
    # when this is set, --data_dir is overrided by local_data_path
    parser.add_argument("--data_url", type=str, default=None)
    parser.add_argument("--train_url", type=str, default=None)
    parser.add_argument('--batch-size', default=4, type=int,
                        metavar='B', help='training batch size')
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--num_pos', default=2, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')

    # image transformer           
    parser.add_argument('--img_w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                        metavar='imgh', help='img height') 
    
    # model
    parser.add_argument('--low-dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
    parser.add_argument('--part', default=3, type=int,
                        metavar='tb', help='part number, either add weighted part attention')
    parser.add_argument('--lambda0', default=1.0, type=float,
                        metavar='lambda0', help='graph attention weights')
    parser.add_argument('--graph', action='store_true', help='either add graph attention or not')

    # loss setting
    parser.add_argument('--method', default='id+tri', type=str,
                        metavar='m', help='method type')
    parser.add_argument('--drop', default=0.2, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--margin', default=0.3, type=float,
                        metavar='margin', help='triplet loss margin')

    # optimizer and scheduler
    parser.add_argument("--lr", default=0.1, type=float, help='learning rate, 0.00035 for adam')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    
    # training configs
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained resnet-50 checkpoint path')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--model_path', default='save_model/', type=str,
                        help='model save path')
    parser.add_argument('--run_distribute', action='store_true', 
                        help="if set true, this code will be run on distrubuted architecture with mindspore")                    
    # parser.add_argument('--debug', action="store_true", 
    #                     help='if set true, use a demo dataset for debugging')
    parser.add_argument('--debug', default="", choices=["yes", "no"],
                            help='if set yes, use a demo dataset for debugging')
    parser.add_argument('--save_epoch', default=20, type=int,
                        metavar='s', help='save model every 10 epochs')
    parser.add_argument('--log_path', default='log/', type=str,
                        help='log save path')
    parser.add_argument('--vis_log_path', default='log/vis_log_ddag/', type=str,
                        help='log save path')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)') 
    parser.add_argument('--seed', default=0, type=int,
                        metavar='t', help='random seed')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--device_target', default="CPU", choices=["CPU","GPU", "Ascend"])
    parser.add_argument('--parameter_server', default=False)
    parser.add_argument('--run_distributed', default=False)
    parser.add_argument('--device_num', default=1)
    

    return parser


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

def decode(img):
    return Image.fromarray(img)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    ########################################################################
    # Init context
    ########################################################################
    # check whether gpu is available 
    target = args.device_target
    if target == "CPU":
        args.run_distribute = False
        context.set_context(mode=context.PYNATIVE_MODE, device_target=target)
    else:
        if args.parameter_server:
            context.set_ps_context(enable_ps=True)
        
        # distributed running context setting
        if args.run_distribute:
            # Ascend target
            if target == "Ascend":
                if args.device_num > 1:
                    # not usefull now, because we only have one Ascend Device
                    pass
            # end of if args.device_num > 1:
                init()
            # GPU target
            else:
                init()
                context.set_auto_parallel_context(
                    device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True
                )
        # end of if target="Ascend":
    # end of if args.run_distribute:

        # Adapt to Huawei Cloud: download data from obs to local location
        if target == "Ascend":
            # Adapt to Cloud: used for downloading data from OBS to docker on the cloud
            import moxing as mox

            local_data_path = "/cache/data"
            args.data_path = local_data_path
            print("Download data...")
            mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_path)
            print("Download complete!(#^.^#)")
            # print(os.listdir(local_data_path))

    ########################################################################
    # Create Dataset
    ########################################################################
    dataset_type = args.dataset

    if dataset_type == "SYSU":
        # TODO: define your data path
        # don't forget add '/' at the end of folder path
        # data_path = './data/SYSU-MM01/'
        data_path = args.data_path
        # data_path = os.path.join(local_data_path, "SYSU-MM01")
        # log_path
        # test_mode = [1,2] # infrared tp visible
    elif dataset_type == "RegDB":
        # TODO: define your data path
        data_path = "./data/REGDB"
        # log_path
        # test_mode = [2,1] # visible tp infared

    
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
            decode,
            py_trans.Pad(10),
            py_trans.RandomCrop((args.img_h, args.img_w)),
            py_trans.RandomHorizontalFlip(),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    ) 

    transform_test = Compose(
        [
            decode,
            py_trans.Resize((args.img_h, args.img_w)),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    ifDebug_dic = {"yes":True, "no":False}
    if dataset_type=="SYSU":
        # train_set
        ifDebug = {}
        trainset_generator = SYSUDatasetGenerator(data_dir=data_path, ifDebug=ifDebug_dic.get(args.debug))
        color_pos, thermal_pos = GenIdx(trainset_generator.train_color_label, trainset_generator.train_thermal_label)
        
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    else:
        pass

    gallset = TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, img_size=(args.img_w, args.img_h))

    print_dataset_info(dataset_type, trainset_generator, query_label, gall_label, start_time)
    
    ########################################################################
    # Define net
    ######################################################################## 
    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)
    net = embed_net(args.low_dim, class_num=n_class, drop=args.drop, part=args.part, arch=args.arch)
    # Print network architecture
    # for m in net.cells_and_names():
    #     print(m)

    # TODO: voncert to mindspore
    # net.to(device) 
    # cudnn.benchmark = True
    
    ########################################################################
    # Define loss
    ######################################################################## 
    criterion1 = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loader_batch = args.batch_size * args.num_pos
    # criterion2 = 

    ########################################################################
    # Define optimizers & schedulers
    ########################################################################
    if args.optim == 'sgd':
        ignored_params = list(map(id, net.bottleneck.get_parameters())) \
                        + list(map(id, net.classifier.get_parameters()))\
                        + list(map(id, net.wpa.get_parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.get_parameters())

        optimizer_P = Momentum([
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
            learning_rate=args.lr, weight_decay=5e-4, momentum=0.9)

    # optimizer_G = optim.SGD([
    #     {'params': net.attention_0.parameters(), 'lr': args.lr},
    #     {'params': net.attention_1.parameters(), 'lr': args.lr},
    #     {'params': net.attention_2.parameters(), 'lr': args.lr},
    #     {'params': net.attention_3.parameters(), 'lr': args.lr},
    #     {'params': net.out_att.parameters(), 'lr': args.lr}, ],
    #     weight_decay=5e-4, momentum=0.9, nesterov=True)    


    ########################################################################
    # Start Training
    ########################################################################
    print('==> Start Training...')
    for epoch in range(start_epoch, 81 - start_epoch):

        print('==> Preparing Data Loader...')
        # identity sampler: 
        sampler = IdentitySampler(trainset_generator.train_color_label, \
                                trainset_generator.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                                epoch)

        trainset_generator.cIndex = sampler.index1 # color index
        trainset_generator.tIndex = sampler.index2 # thermal index

        # add sampler
        trainset = ds.GeneratorDataset(trainset_generator, ["color", "thermal","color_label", "thermal_label"],sampler=sampler).map(
            operations=transform_train, input_columns=["color"])

        # remove sampler(although it disagrees with original paper implementation)
        # trainset = ds.GeneratorDataset(trainset_generator, ["color", "thermal","color_label", "thermal_label"]).map(
        #     operations=transform_train, input_columns=["color"])

        trainset = trainset.map(operations=transform_train, input_columns=["thermal"])

        
        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # infrared index
        print(epoch)
        # print(sampler.cIndex)
        # print(sampler.tIndex)

        loader_batch = args.batch_size * args.num_pos

        # define callbacks
        loss_cb = LossMonitor()
        cb = [loss_cb]

        trainset = trainset.batch(batch_size=loader_batch)
        dataset_helper = DatasetHelper(trainset, dataset_sink_mode=False)
        # for inputs in dataset_helper:
        #     print(*inputs)

        # net_with_criterion = WithLossCell(net, criterion1)
        net_with_criterion = MyWithLossCell(net, criterion1)
        net_with_optim = TrainOneStepCell(net_with_criterion, optimizer_P)
        # train_net = connect_network_with_dataset(net_with_optim, dataset_helper) # only work in GPU/Ascend Mode

        for (img1, img2, label1, label2) in dataset_helper:
            print("img1 shape is :", img1.shape)
            print("label1 is ", label1)
            net_with_optim.set_train()
            label1 = ms.Tensor(label1, dtype=ms.float32)
            label2 = ms.Tensor(label2, dtype=ms.float32)
            loss = net_with_optim(img1, img2, label1, label2)
            print("loss is :", loss)
            print("*******************")

        # model = Model(net, loss_fn=criterion1, optimizer=optimizer_P, metrics=None)
        # model.train(1, trainset, callbacks=cb)



        
