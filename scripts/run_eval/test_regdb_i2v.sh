export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

myfile="test_regdb_i2v.sh"

if [! -f "$myfile"]; then
    echo "Please first enter DDAG_mindspore/scripts/run_eval and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset RegDB \
--gpu 2 \
--device-target GPU \
--resume "/home/shz/pytorch/zzw/DDAG_mindspore_modify/logs/regdb_i2v/training/epoch_35_rank1_84.17_mAP_78.43_RegDB_batch-size_2*8*4=64_adam_lr_0.0035_loss-func_id+tri_Graph__trial_1_main.ckpt" \
--tag "regdb_i2v" \
--data-path "/home/shz/pytorch/data/regdb" \
--branch main \
--regdb-mode "i2v" \
--part 0