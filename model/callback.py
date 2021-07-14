import mindspore as ms
from mindspore.train import callback
from utils.utils import *

def adjust_learning_rate(optimizer_P, optimizer_G=None, epoch=0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 20:
        lr = args.lr
    elif 20 <= epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr

class PreEpoch(callback):
    def __init__(self, run_context):
        super(PreEpoch, self).__init__()

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        opt = cb_params.optimizer
        epoch_num = cb_params.cur_epoch_num
        batch_num = cb_params.cur_step_num
        adjust_learning_rate(opt, epoch=epoch_num) # TODO: check whether the optimizer in model can be altered
        
        

class PreStep(callback):
    def __init__(self, run_context):
        super(PreStep, self).__init__()

    def step_begin(self, run_context):
        train_loss = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()
        # graph_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0


