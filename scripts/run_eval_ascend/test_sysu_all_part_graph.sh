#!/usr/bin/env bash
myfile="test_sysu_all_part_graph.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset SYSU \
--device-id 0 \
--device-target Ascend \
--resume "XXX.ckpt" \
--tag "sysu_all_part_graph" \
--data-path "Define your own path/sysu" \
--branch main \
--sysu-mode "all" \
--part 3