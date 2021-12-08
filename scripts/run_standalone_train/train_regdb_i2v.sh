export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

myfile="train_regdb_i2v.sh"

if [ ! -f "$myfile"]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--dataset RegDB \
--optim adam \
--lr 0.0035 \
--gpu 0 \
--device-target GPU \
--pretrain "resnet50.ckpt" \
--tag "regdb_i2v" \
--data-path "/home/shz/pytorch/data/regdb" \
--loss-func "id+tri" \
--branch main \
--regdb-mode "i2v" \
--part 0 \
--graph