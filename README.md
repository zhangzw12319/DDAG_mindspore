# DDAG: Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification

Mindspore implementation for ***Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification*** in ECCV 2020. Please read the [original paper](https://arxiv.org/pdf/2007.09314.pdf)  or original [pytorch implementation](https://github.com/mangye16/DDAG) for a more detailed description of the training procedure.

## Results

### SYSU-MM01 (all-search mode)

| Metric | Value(Pytorch) | Value(Mindspore) |
| :----: | :------------: | :--------------: |
| Rank-1 |     54.75%     |      58.75%      |
|  mAP   |     53.02%     |      56.36%      |

### SYSU-MM01 (indoor-search mode)

| Metric | Value(Pytorch) | Value(Mindspore) |
| :----: | :------------: | :--------------: |
| Rank-1 |     61.02%     |      65.46%      |
| Rank-1 |     67.98%     |      69.28%      |

### RegDB(Visible-Thermal)

| Metric | Value(Pytorch) | Value(Mindspore) |
| :----: | :------------: | :--------------: |
| Rank-1 |     69.34%     |      85.49%      |
|  mAP   |     63.46%     |      80.01%      |

### RegDB(Thermal-Visible)

| Metric | Value(Pytorch) | Value(Mindspore) |
| :----: | :------------: | :--------------: |
| Rank-1 |     68.06%     |      84.17%      |
|  mAP   |     61.80%     |      78.43%      |

***Note**: The aforementioned pytorch results can be seen in original pytorch repo.

## Requirements

- Python==3.7.5
- Mindspore==1.5.0(See [Installation](https://www.mindspore.cn/install/))
- Cuda==10.1

## Get Started

### 1. Prepare the datasets

- (1) SYSU-MM01 Dataset [1]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).
- run `python pre_process_sysu.py` [link](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/blob/master/pre_process_sysu.py) in to prepare the dataset, the training data will be stored in ".npy" format.
- (2) RegDB Dataset [2]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.
- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website).

### 2. Pretrain Model

We use ImageNet-2012 dataset to pretrain resnet-50 backbone. For copyright reasons, checkpoint files cannot be shown publicly. For those who are interesting in our mindspore model, you can get access to  the checkpoint files for academic use only. Please contact zhangzw12319@163.com for application.

Furthermore, our model can still be trained without pretraining. It may lose about  4%-5% precision in CMC & mAP compared to pretrained ones at same training epochs.

### 3. Training

Train a model by

```bash
python train.py --dataset SYSU --data-path "Path-to-dataset" --optim adam --lr 0.0035 --device-target GPU --gpu 0 --pretrain "Path-to-pretrain-model.ckpt" --part 3 --graph
```

- `--dataset`: which dataset "SYSU" or "RegDB".
- `--data-path`: manually define the data path(for `SYSU`, path folder must contain `.npy` files, see [`pre_process_sysu.py`](#anchor1) ).
- `--optim`: choose "adam" or "sgd"(default adam)
- `--lr`: initial learning rate( 0.0035 for adam, 0.1 for sgd)
- `--device-target` choose "GPU" or "Ascend"(TODO: add Huawei Model Arts support)
- `--gpu`: which gpu to run(default: 0)
- `--pretrain`: specify resnet-50 checkpoint file path(default "" for no ckpt file)
- `--resume`: specify checkpoint file path for whole model(default "" for no ckpt file, `--resume` loads weights after `--pretrain`, and thus will overwrite `--pretrain` weights). **Please note that mindspore compulsorily requires checkpoint files have `.cpkt` as file suffix, otherwise may trigger errors during loading.**
- `--tags`(optional): You can name(tag) your experiments to organize you logs file, e.g. specifying `--tag Exp_1` will put your log files into "logs/Exp_1/XXX".Default value is "toy", which means toy experiments(e.g. debugging logs).

Other useful arguments:

We recommend that these hyper-parameters should be kept by default to achieve SOTA performance as reported in the paper. If you want to fine-tune these hyper-parameters to get a better performance, just feel free to modify them.

- `--msmode`: Mindspore running mode, either 'GRAPH_MODE' or 'PYNATIVE_MODE'.
- `--epoch`: the total number of training epochs, by default 40 Epochs(may be different from original paper).
- `--warmup-steps`: by default 5
- `--start-decay`: the start epoch of lr decay, by default 15.
- `--end-decay`: the ending epoch of lr decay , by default 27.
- `--loss-func`: for ablation study, by default "id+tri" which is cross-entropy loss plus triplet loss. You can choose from `["id", "id+tri"]`.
- `--sysu-mode` choose from `["all", "indoor"]`
- `--regdb-mode` choose from `["i2v", "v2i"]`
- `--graph` if set `--graph` then use graph attention module
- `--part` if set`--part X` , then use local part attention module with part number X; if you don't want to use this module, you can set `--part 0`

### 4. Evaluation

For now, we integrate evaluation module into train.py. Every epoch we will do evaluations.

Furthermore, you can directly run test.py, with following format

```bash
python test.py --dataset SYSU --data-path "Path-to-dataset" --device-target GPU --gpu 0 --resume "Path-to-saved-checkpoint-file.ckpt" --part 3 --graph
```

## Citation

Please kindly cite the original paper references in your publications if it helps your research:

> @inproceedings{eccv20ddag,
> title={Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification},
> author={Ye, Mang and Shen, Jianbing and Crandall, David J. and Shao, Ling and Luo, Jiebo},
> booktitle={European Conference on Computer Vision (ECCV)},
> year={2020},
> }

Please kindly reference the url of mindspore repository in your code if it helps your research and code:

> `https://github.com/zhangzw12319/DDAG_mindspore` or
> `https://gitee.com/zhangzw12319/DDAG_mindspore`

## TODO

### Key fixes

-

### Other fixes

- add Huawei Model Arts support
