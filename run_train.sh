#!/bin/sh
#PBS -q week
#PBS -l nodes=1:ppn=32:gpus=1
#PBS -l walltime=168:00:00

test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
echo `hostname`

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/kukai/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/kukai/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/kukai/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/kukai/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd /home/kukai/kaku/pneumonia-efficientnet

conda activate tf_env

python /home/kukai/kaku/pneumonia-efficientnet/models/train.py

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True




