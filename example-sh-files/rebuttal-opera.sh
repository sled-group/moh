#!/bin/bash
#SBATCH -A uva_cv_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rqa8sm@virginia.edu
#SBATCH --output=/scratch/rqa8sm/rebuttal/code/moh/example-sh-files/output-rebuttal/output-1.txt
#SBATCH --error=/scratch/rqa8sm/rebuttal/code/moh/example-sh-files/output-rebuttal/err-1.err

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
conda activate /scratch/rqa8sm/rebuttal/env/opera

# Run the Python script
python main.py --model_name operallava --model_size "7b" --model_path /scratch/rqa8sm/rebuttal/data/llava-v1.5-7b --device_map cuda --data_base_path /scratch/rqa8sm/rebuttal/data/ROPE --output_base_path /scratch/rqa8sm/rebuttal/result
