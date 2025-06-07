#! /bin/bash

#SBATCH -t 24:00:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes 1

cd ~/desi-misc

(echo "legacypipe:";
 (cd ~/legacypipe && git describe);  #  --> DR10.3.3-72-g3b8db910
 echo "desimodel-data:";
 (cd ~/desimodel-data && svn up);    #  --> Updated to revision 144384.
 echo "desi-misc:";
 git describe) > stuck.log

module unload tractor
unset PYTHONPATH
source /global/common/software/desi/desi_environment.sh 24.4
module use /global/common/software/desi/users/dstn/modulefiles/
module load tractor/desi24.4
export PYTHONPATH=${PYTHONPATH}:~/legacypipe/py
python -u stuck-positioners-on-bright-stars.py >> stuck.log 2>&1

