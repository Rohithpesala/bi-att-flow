#!/bin/bash
#
#SBATCH --job-name=1-gpu-bidaf-tf
#SBATCH --partition=m40-short
#SBATCH --output=bidaf-m40s-1-%A.out
#SBATCH --error=bidaf-m40s-1-%A.err
#SBATCH --gres=gpu:1

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

#module purge
module load python/3.5.2
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44

# Remove any previous versions of Tensorflow 
pip uninstall tensorflow

## Dependencies
pip install --user tqdm nltk jinja2
# USE TF R0.11 ONLY ## TF SUCKS
pip install --user --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl


## Change this line so that it points to your bidaf gitbuh folder
cd /home/usaxena/work/s18/696/bidaf-tf/bi-att-flow

## Change lines below if you want to run it differently

## Debugging a small model - use this to test if any changes in your code 
python -m basic.cli --mode train --noload --debug  --batch_size 60 --device /gpu:0 --device_type gpu --num_gpus 1 --len_opt --cluster

## Train Full dataset
# python -m basic.cli --mode train --noload --batch_size 60 --num_steps 10000 -device /gpu:0 --device_type gpu --num_gpus 1 --len_opt --cluster

## Testing on dataset
# python -m basic.cli -device /gpu:0 --device_type gpu --num_gpus 1 --len_opt --cluster
