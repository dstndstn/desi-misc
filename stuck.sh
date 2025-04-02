#! /bin/bash

#SBATCH -t 24:00:00
#SBATCH -C cpu
#SBATCH --nodes 1

cd ~/desi-misc
module unload tractor
unset PYTHONPATH
source /global/common/software/desi/desi_environment.sh 24.4
module use /global/common/software/desi/users/dstn/modulefiles/
module load tractor/desi24.4
export PYTHONPATH=${PYTHONPATH}:~/legacypipe/py
python -u stuck-positioners-on-bright-stars.py > stuck.log 2>&1

