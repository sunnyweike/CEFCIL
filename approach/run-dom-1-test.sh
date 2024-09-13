#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-13,paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4
module load anaconda/2021.11  compilers/gcc/9.3.0  compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x  
source activate pytorch
export PYTHONUNBUFFERED=1
python src-0418-0/main_incremental.py --approach ILFTF --gmms 1 --max-experts 5 --use-multivariate --nepochs 200 --ftepochs 100 --tau 3 --batch-size 128 --num-workers 4 --datasets domainnet  --num-tasks 36  --nc-first-task 5 --lr 0.05 --weight-decay 1e-3 --clipping 1 --alpha 0.99 --use-test-as-val --network resnet18 --momentum 0.9 --exp-name exp_1_0419-0 --seed 0 --small_data_rate 0.1 --num_ini_classes 90  --num_experts 5 --num_ini_tasks 18 --random_seed 0 --collapse_alpha 0 --inc_sub_rate 0.9 --drift_loss_threshold 1 --window_size 10

