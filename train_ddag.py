# DDAG Mindspore version(2021.07)
# Developer List:
# [@zhangzw12319](https://github.com/zhangzw12319)
# [@sunhz0117](https://github.com/sunhz0117)
# Pytorch Code & Original Paper from https://github.com/mangye16/DDAG


import os
import os.path as osp
import sys
import time

import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.vision.py_transforms as py_trans

from mindspore import context, Model, load_checkpoint, load_param_into_net, save_checkpoint, DatasetHelper, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.train.callback import LossMonitor
from mindspore.train.dataset_helper import connect_network_with_dataset
from mindspore.nn import SGD, Adam, TrainOneStepCell, WithLossCell


from data.data_loader import *
from data.data_manager import *
from model.eval import test
from model.lr_generator import LR_Scheduler
from model.model_main import *
from model.trainingCell import Criterion_with_Net, Optimizer_with_Net_and_Criterion
from model.resnet import *
from utils.loss import *
from utils.utils import *

from PIL import Image
from IPython import embed
from tqdm import tqdm


def show_memory_info(hint=""):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")


def get_parser():
    parser = argparse.ArgumentParser(description="DDAG Code Mindspore Version")

    # dataset settings
    parser.add_argument("--dataset", default='SYSU', choices=['SYSU', 'RegDB'],
                        help='dataset name: RegDB or SYSU')
    parser.add_argument('--data-path',type=str, default='data' )
    # Only used on Huawei Cloud OBS service,
    # when this is set, --data_path is overrided by --data-url
    parser.add_argument("--data-url", type=str, default=None)
    parser.add_argument('--batch-size', default=8, type=int,
                        metavar='B', help='the number of person IDs in a batch')
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--num-pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--debug', default="no", choices=["yes", "no"],
                        help='if set yes, use demo dataset for debugging,(only for SYSU dataset)')
    # TODO: add multi worker dataloader support
    # parser.add_argument('--workers', default=4, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')


    # image transform         
    parser.add_argument('--img-w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img-h', default=288, type=int,
                        metavar='imgh', help='img height') 
    

    # model
    parser.add_argument('--low-dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
    parser.add_argument('--part', default=0, type=int,
                        metavar='tb', help='part number, either add weighted part attention  module')
    # TODO: add multi head graph attention module
    # parser.add_argument('--lambda0', default=1.0, type=float,
    #                     metavar='lambda0', help='graph attention weights')
    parser.add_argument('--graph', action='store_true', help='either add graph attention or not')


    # loss setting
    parser.add_argument('--epoch', default=80, type=int,
                        metavar='epoch', help='epoch num')
    parser.add_argument('--loss-func', default='id+tri', type=str, choices=['id', 'tri', 'id+tri'],
                        metavar='m', help='specify loss fuction type')
    parser.add_argument('--drop', default=0.2, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--margin', default=0.3, type=float,
                        metavar='margin', help='triplet loss margin')


    # optimizer and scheduler
    parser.add_argument("--lr", default=0.0035, type=float, help='learning rate, 0.0035 for adam; 0.1 for sgd')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer')
    parser.add_argument("--warmup-steps", default=5, type=int, help='warmup steps')
    
    # training configs
    parser.add_argument('--device-target', default="CPU", choices=["CPU","GPU", "Ascend"])
    parser.add_argument('--gpu', default='0', type=str, help='set CUDA_VISIBLE_DEVICES')

    # Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU).
    #  If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be 
    # the number set in the environment variable 'CUDA_VISIBLE_DEVICES'.
    #  For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 
    # 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
    parser.add_argument('--device-id', default=0, type=str, help='')

    parser.add_argument('--device-num', default=1, type=int, help='the total number of available gpus')
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint, no resume:""')
    parser.add_argument('--pretrain', type=str, default="model/pretrain/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt",
                        help='Pretrain resnet-50 checkpoint path, no pretrain: ""')
    parser.add_argument('--model-path', default='ckpt/', type=str,
                        help='model checkpoint save path')
    parser.add_argument('--run_distribute', action='store_true', 
                        help="if set true, this code will be run on distrubuted architecture with mindspore")                    
    parser.add_argument('--parameter-server', default=False)
    

    # logging configs
    parser.add_argument('--ckpt', default='test', type=str, help='ckpt suffix name')
    parser.add_argument("--branch-name", default="master",
                        help="Github branch name, for ablation study tagging")
    
    # testing / evaluation config 
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    
    return parser


def print_dataset_info(dataset_type, trainset, query_label, gall_label, start_time):
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


def optim(epoch, backbone_lr_scheduler, head_lr_scheduler):
    ########################################################################
    # Define optimizers
    ########################################################################
    epoch = ms.Tensor(epoch, ms.int32)
    backbone_lr = float(backbone_lr_scheduler(epoch).asnumpy())
    head_lr = float(head_lr_scheduler(epoch).asnumpy())

    if args.optim == 'sgd':
        ignored_params = list(map(id, net.bottleneck.trainable_params())) \
                       + list(map(id, net.classifier.trainable_params())) \
                        # + list(map(id, net.wpa.trainable_params())) \

        base_params = list(filter(lambda p: id(p) not in ignored_params, net.trainable_params()))

        optimizer_P = SGD([
            {'params': base_params, 'lr': backbone_lr},
            {'params': net.bottleneck.trainable_params(), 'lr': head_lr},
            {'params': net.classifier.trainable_params(), 'lr': head_lr},
            # {'params': net.wpa.trainable_params(), 'lr': head_lr},
            ],
            learning_rate=args.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)

    elif args.optim == 'adam':
        ignored_params = list(map(id, net.bottleneck.trainable_params())) \
                       + list(map(id, net.classifier.trainable_params())) \
                    #    + list(map(id, net.wpa.trainable_params())) \

        base_params = list(filter(lambda p: id(p) not in ignored_params, net.trainable_params()))

        optimizer_P = Adam([
            {'params': base_params, 'lr': backbone_lr},
            {'params': net.bottleneck.trainable_params(), 'lr': head_lr},
            {'params': net.classifier.trainable_params(), 'lr': head_lr},
            # {'params': net.wpa.trainable_params(), 'lr': head_lr},
        ],
            learning_rate=args.lr, weight_decay=5e-4)

    return optimizer_P


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.device_target == 'GPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ########################################################################
    # Init context
    ########################################################################
    device = args.device_target
    # init context
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device, save_graphs=False)

    if device == "CPU":
        local_data_path = args.data_path
        args.run_distribute = False
    else:
        if device == "GPU":
            local_data_path = args.data_path
            context.set_context(device_id=args.device_id)

        if args.parameter_server:
            context.set_ps_context(enable_ps=True)
        
        # distributed running context setting
        if args.run_distribute:
            # Ascend target
            if device == "Ascend":
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
                # mixed precision setting
                context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        # end of if target="Ascend":
    # end of if args.run_distribute:

        # Adapt to Huawei Cloud: download data from obs to local location
        if device == "Ascend":
            # Adapt to Cloud: used for downloading data from OBS to docker on the cloud
            import moxing as mox

            local_data_path = "/cache/data"
            args.data_path = local_data_path
            print("Download data...")
            mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_path)
            print("Download complete!(#^.^#)")
            # print(os.listdir(local_data_path))


    ########################################################################
    # Logging
    ########################################################################
    loader_batch = args.batch_size * args.num_pos

    if device == "GPU" or device == "CPU":
        checkpoint_path = os.path.join("ckpt", args.ckpt)
        os.makedirs(checkpoint_path, exist_ok=True)

        suffix = args.ckpt + "_" + str(args.dataset)
        
        suffix = suffix + '_batch-size_2*{}*{}={}'.format(args.batch_size, args.num_pos, 2 * loader_batch)
        suffix = suffix + '_{}_lr_{}'.format(args.optim, args.lr)
        suffix = suffix + '_loss-func_{}'.format(args.loss_func)

        if args.part > 0:
            suffix = suffix + '_P_{}'.format(args.part)
        
        if args.dataset == 'RegDB':
            suffix = suffix + '_trial_{}'.format(args.trial)

        suffix = suffix + "_" + args.branch_name

        time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = open(osp.join(checkpoint_path, "{}_performance_{}.txt".format(suffix, time_msg)), "w")
        # error_file = open(osp.join(checkpoint_path, "{}_error_{}.txt".format(suffix, time_msg)), "w")
        # pretrain_file = open(osp.join(checkpoint_path, "{}_pretrain_{}.txt".format(suffix, time_msg)), "w")
        print('Args: {}'.format(args))
        print('Args: {}'.format(args), file=log_file)


    ########################################################################
    # Create Dataset
    ########################################################################
    dataset_type = args.dataset

    if dataset_type == "SYSU":
        # infrared to visible(1->2)
        # TODO: define your data path
        data_path = osp.join(args.data_path, 'sysu')
    elif dataset_type == "RegDB":
        # visible tu infrared(2->1)
        # TODO: define your data path
        data_path = osp.join(args.data_path, 'regdb')
    
    best_acc = 0
    best_acc = 0  # best test accuracy
    start_epoch = 1
    feature_dim = args.low_dim
    wG = 0
    start_time = time.time()

    print("==> Loading data")
    # Data Loading code

    transform_train_rgb = Compose(
        [
            decode,
            # py_trans.Pad(10),
            # py_trans.RandomCrop((args.img_h, args.img_w)),
            py_trans.RandomGrayscale(prob=0.5),
            py_trans.RandomHorizontalFlip(),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            py_trans.RandomErasing(prob=0.5)
        ]
    ) 

    transform_train_ir = Compose(
        [
            decode,
            # py_trans.Pad(10),
            # py_trans.RandomCrop((args.img_h, args.img_w)),
            # py_trans.RandomGrayscale(prob=0.5),
            py_trans.RandomHorizontalFlip(),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            py_trans.RandomErasing(prob=0.5)
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
    
    ifDebug_dic = {"yes": True, "no": False}
    if dataset_type == "SYSU":
        # train_set
        trainset_generator = SYSUDatasetGenerator(data_dir=data_path, transform_rgb=transform_train_rgb, transform_ir=transform_train_ir,  ifDebug=ifDebug_dic.get(args.debug))
        color_pos, thermal_pos = GenIdx(trainset_generator.train_color_label, trainset_generator.train_thermal_label)
        
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    elif dataset_type == "RegDB":
        pass

    ########################################################################
    # Create Query && Gallery
    ########################################################################

    gallset_generator = TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h), transform=transform_test)
    queryset_generator = TestData(query_img, query_label, img_size=(args.img_w, args.img_h), transform=transform_test)

    print_dataset_info(dataset_type, trainset_generator, query_label, gall_label, start_time)
    
    ########################################################################
    # Define net
    ######################################################################## 
    
    # pretrain
    if len(args.pretrain) > 0:
        print("Pretrain model: {}".format(args.pretrain))
        print("Pretrain model: {}".format(args.pretrain), file=log_file)

    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    net = embed_net(args.low_dim, class_num=n_class, drop=args.drop, part=args.part, arch=args.arch, pretrain=args.pretrain)

    if len(args.resume) > 0:
        print("Resume checkpoint:{}". format(args.resume))
        print("Resume checkpoint:{}". format(args.resume), file=log_file)
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(net, param_dict)
        if args.resume.split("/")[-1].split("_")[0] != "best":
            args.resume = int(args.resume.split("/")[-1].split("_")[1])
        print("Start epoch: {}".format(args.resume))
        print("Start epoch: {}".format(args.resume), file=log_file)
        

    ########################################################################
    # Define loss
    ######################################################################## 
    CELossNet = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    OriTripLossNet = OriTripletLoss(margin=args.margin, batch_size=2 * loader_batch)
    # TripLossNet = TripletLoss(margin=args.margin)

    net_with_criterion = Criterion_with_Net(net, CELossNet, OriTripLossNet, lossFunc=args.loss_func)

    ########################################################################
    # Define schedulers
    ########################################################################

    backbone_lr_scheduler = LR_Scheduler(0.1 * args.lr, args.warmup_steps, [15, 27])
    head_lr_scheduler = LR_Scheduler(args.lr, args.warmup_steps, [15, 27])

    ########################################################################
    # Start Training
    ########################################################################

    print('==> Start Training...')
    best_mAP = 0.0
    best_r1 = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, args.epoch + 1):

        optimizer_P = optim(epoch, backbone_lr_scheduler, head_lr_scheduler)
        net_with_optim = Optimizer_with_Net_and_Criterion(net_with_criterion, optimizer_P)

        print('==> Preparing Data Loader...')
        # identity sampler: 
        sampler = IdentitySampler(trainset_generator.train_color_label,
                                trainset_generator.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size)

        trainset_generator.cIndex = sampler.index1  # color index
        trainset_generator.tIndex = sampler.index2  # thermal index

        # add sampler
        trainset = ds.GeneratorDataset(trainset_generator, ["color", "thermal", "color_label", "thermal_label"],
                                       sampler=sampler)

        trainset = trainset.map(operations=transform_train_rgb, input_columns=["color"])
        trainset = trainset.map(operations=transform_train_ir, input_columns=["thermal"])


        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # infrared index
        print("Epoch [{}]".format(str(epoch)))

        # define callbacks
        loss_cb = LossMonitor()
        cb = [loss_cb]

        trainset = trainset.batch(batch_size=loader_batch, drop_remainder=True)

        dataset_helper = DatasetHelper(trainset, dataset_sink_mode=False)
        # net_with_optim = connect_network_with_dataset(net_with_optim, dataset_helper)

        net.set_train(mode=True)
      
        batch_idx = 0
        N = np.maximum(len(trainset_generator.train_color_label), len(trainset_generator.train_thermal_label))
        total_batch = int(N / loader_batch) + 1
        print("The total number of batch is ->", total_batch)

        # calculate average batch time
        batch_time = AverageMeter()
        end_time = time.time()
        
        # calculate average accuracy
        acc = AverageMeter()

        
        for batch_idx, (img1, img2, label1, label2) in enumerate(tqdm(dataset_helper)):
            label1, label2 = ms.Tensor(label1, dtype=ms.float32), ms.Tensor(label2, dtype=ms.float32)
            img1, img2 = ms.Tensor(img1, dtype=ms.float32), ms.Tensor(img2, dtype=ms.float32)

            loss = net_with_optim(img1, img2, label1, label2)
            acc.update(net_with_criterion.acc)
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if batch_idx % 100 == 0:
                print('Epoch: [{}][{}/{}]   '
                      'LR: {LR:.4f}   '
                      'Loss:{Loss:.4f}   '
                    #   'id:{Loss:.4f}   '
                    #   'tri:{Loss:.4f}   '
                      'Batch Time:{batch_time:.2f}  '
                      'Accuracy:{acc:.2f}   '
                      .format(epoch, batch_idx, total_batch,
                              LR=float(head_lr_scheduler(ms.Tensor(epoch, ms.int32)).asnumpy()),
                              Loss=float(loss.asnumpy()),
                            #   id=float(loss_dict["id"].asnumpy()),
                            #   tri=float(loss_dict["tri"].asnumpy()),
                              batch_time=batch_time.avg,
                              acc = acc.avg * 100
                              ))
                print('Epoch: [{}][{}/{}]   '
                      'LR: {LR:.4f}   '
                      'Loss:{Loss:.4f}   '
                      'Batch Time:{batch_time:.3f}  '
                      'Accuracy:{acc:.4f}   '
                      .format(epoch, batch_idx, total_batch,
                              LR=float(head_lr_scheduler(ms.Tensor(epoch, ms.int32)).asnumpy()),
                              Loss=float(loss.asnumpy()),
                              batch_time=batch_time.avg,
                              acc = acc.avg * 100
                              ), file=log_file)
        show_memory_info("train: epoch {}".format(epoch))

        if epoch > 0:

            net.set_train(mode=False)
            gallset = ds.GeneratorDataset(gallset_generator, ["img", "label"])
            gallset = gallset.map(operations=transform_test, input_columns=["img"])
            gallery_loader = gallset.batch(batch_size=args.test_batch)
            gallery_loader = DatasetHelper(gallery_loader, dataset_sink_mode=False)

            queryset = ds.GeneratorDataset(queryset_generator, ["img", "label"])
            queryset = queryset.map(operations=transform_test, input_columns=["img"])
            query_loader = queryset.batch(batch_size=args.test_batch)
            query_loader = DatasetHelper(query_loader, dataset_sink_mode=False)

            if args.dataset == "SYSU":
                # import pdb
                # pdb.set_trace()
                cmc, mAP, cmc_att, mAP_att = test(args, gallery_loader, query_loader, ngall,
                    nquery, net, 1, gallery_cam=gall_cam, query_cam=query_cam)
            
            if args.dataset == "RegDB":
                cmc, mAP, cmc_att, mAP_att = test(args, gallset, queryset, ngall,
                    nquery, net, 2)

            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=log_file)

            if args.part > 0:
                print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                    cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att))
                print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                    cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att), file=log_file)

            if mAP > best_mAP:
                best_mAP = mAP

            if cmc[0] > best_r1:
                path = osp.join(checkpoint_path, f"epoch_{epoch:02}_rank1_{cmc[0]*100:.2f}_mAP_{mAP*100:.2f}_{suffix}.ckpt")
                save_checkpoint(net, path)
                path = osp.join(checkpoint_path, f"best_{suffix}.ckpt")
                save_checkpoint(net, path)

                best_r1 = cmc[0]
                best_epoch = epoch

            print('Best(Epoch {}):   Rank-1: {:.2%} | mAP: {:.2%}'.format(best_epoch, best_r1, best_mAP))
            print('Best(Epoch {}):   Rank-1: {:.2%} | mAP: {:.2%}'.format(best_epoch, best_r1, best_mAP), file=log_file)

            print("*****************************************************************************************")
            print("*****************************************************************************************", file=log_file)

            log_file.flush()

        show_memory_info("test: epoch {}".format(epoch))

    print(f"Best mAP: {best_mAP:.4f}, Best rank-1: {best_r1:.4f}, Best epoch: {best_epoch}(according to Rank-1)")
    print(f"Best mAP: {best_mAP:.4f}, Best rank-1: {best_r1:.4f}, Best epoch: {best_epoch}(according to Rank-1)", file=log_file)
    log_file.flush()
    log_file.close()
