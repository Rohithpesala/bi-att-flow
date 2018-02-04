#!/bin/bash
#
#SBATCH --job-name=bidaf-tf
#SBATCH --partition=titanx-long
#SBATCH --output=bidaf-tf-experiment_%A.out
#SBATCH --error=bidaf-tf-experiment_%A.err
#SBATCH --gres=gpu:4

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-lstm-jobs.txt

module purge
module load python/3.5.2
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44

pip uninstall --user tensorflow
pip install --user tqdm nltk jinja2

pip install --user --ignore-installed tensorflow-gpu==1.2.0

cd /home/usaxena/work/s18/696/bidaf-tf/bi-att-flow

python -m basic.cli --mode train --noload
