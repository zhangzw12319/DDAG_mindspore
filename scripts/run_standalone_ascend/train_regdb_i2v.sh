smyfile="train_regdb_i2v.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--dataset RegDB \
--optim adam \
--lr 0.0035 \
--device-id 1 \
--device-target Ascend \
--pretrain "resnet50.ckpt" \
--tag "regdb_i2v" \
--data-path "" \
--loss-func "id+tri" \
--branch main \
--regdb-mode "i2v" \
--part 0 \
--graph