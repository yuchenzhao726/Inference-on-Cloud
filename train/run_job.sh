#!/bin/sh

current_dir=$(cd $(dirname $0); pwd)
echo "current_dir $current_dir"

# which pip
# python -m pip install -r $current_dir/requirements.txt

python $current_dir/main.py --num-workers 4

run_current_dir=$(pwd)
echo "clean $run_current_dir"
rm -fr $run_current_dir/cml_proj2