#!/bin/bash
#
#SBATCH --mem=20000
#SBATCH --job-name=1-gpu-bidaf-tf
#SBATCH --partition=m40-short
#SBATCH --output=bidaf-m40s-1-%A.out
#SBATCH --error=bidaf-m40s-1-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruppaal@cs.umass.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.5.2
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44
cd ../..


## Debug
# python -m basic.cli --mode train --noload --debug  --device /gpu:0 --device_type gpu --num_gpus 1

## Train Full dataset
python -m basic.cli --mode train --noload --num_steps 20000 -device /gpu:0 --device_type gpu --num_gpus 1 --save_period 10  --data_dir "data/marco"

## Testing on dataset
python -m basic.cli -device /gpu:0 --device_type gpu --num_gpus 1 
