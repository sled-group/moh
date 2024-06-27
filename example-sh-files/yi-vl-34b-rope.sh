#!/bin/bash
# fill your credential
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:2
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4

module load cuda
cd /scratch/rqa8sm/ROPE/moh

# Initialize conda
__conda_setup="$('/apps/software/standard/core/anaconda/2023.07-py3.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/software/standard/core/anaconda/2023.07-py3.11/etc/profile.d/conda.sh" ]; then
        . "/apps/software/standard/core/anaconda/2023.07-py3.11/etc/profile.d/conda.sh"
    else
        export PATH="/apps/software/standard/core/anaconda/2023.07-py3.11/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate the conda environment
conda activate /scratch/rqa8sm/ROPE/rope_env

# Run the Python script
python main.py --model_name yivl --model_size "34b" --model_path /scratch/rqa8sm/ROPE/Yi-VL-34B-hf --device_map balanced --data_base_path /scratch/rqa8sm/ROPE/ROPE --output_base_path /scratch/rqa8sm/ROPE/output
