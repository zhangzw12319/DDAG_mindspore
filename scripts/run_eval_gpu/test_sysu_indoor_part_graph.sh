export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

myfile="test_sysu_indoor_part_graph.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset SYSU \
--gpu 0 \
--device-target GPU \
--resume "XXX.ckpt" \
--tag "sysu_indoor_part_graph" \
--data-path "Define your own path/sysu" \
--branch main \
--sysu-mode "indoor" \
--part 3