# desi-misc
Miscellaneous tasks for DESI


Nudge tile centers to avoid bright stars landing on stuck fiber positioners.

## 2025-04-01

Running `stuck-positioners-on-bright-stars.py`:

# On a Perlmutter interactive node:
(this timed out... instead have to `sbatch stuck.sh`)

module unload tractor
unset PYTHONPATH
source /global/common/software/desi/desi_environment.sh 24.4
module use /global/common/software/desi/users/dstn/modulefiles/
module load tractor/desi24.4
export PYTHONPATH=${PYTHONPATH}:~/legacypipe/py
(cd ~/legacypipe && git describe)  #  --> DR10.3.3-72-g3b8db910
(cd ~/desimodel-data && svn up)    #  --> Updated to revision 144384.
python -u stuck-positioners-on-bright-stars.py > log 2>&1 &

## 2023-11-09
stuck-positioners-on-bright-stars.py

unset PYTHONPATH
source /global/common/software/desi/desi_environment.sh 23.1
module use /global/common/software/desi/users/dstn/modulefiles/
module load tractor/desi23.1
export PYTHONPATH=${PYTHONPATH}:~/legacypipe/py
python -u stuck-positioners-on-bright-stars.py > log 2>&1 &

## 2023-03-21
stuck-positioners-on-bright-stars.py

unset PYTHONPATH
source /global/common/software/desi/desi_environment.sh 22.5
export PYTHONPATH=${PYTHONPATH}:~/astrometry-desi:~/tractor:~/legacypipe/py
python -u stuck-positioners-on-bright-stars.py > log 2>&1 &

