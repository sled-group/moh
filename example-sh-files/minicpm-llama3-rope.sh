#!/bin/bash
# fill your credential
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email>
#SBATCH --output=<your-output-repo-place>
#SBATCH --error=<your--error-repo-place>

module load cuda
cd /<your-path>/ROPE/moh

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
python main.py --model_name minicpm --model_size "7b" --model_path /<your-path>/ROPE/MiniCPM-Llama3-V-2_5 --device_map cuda --data_base_path /<your-path>/ROPE/ROPE --output_base_path /<your-path>/ROPE/output
