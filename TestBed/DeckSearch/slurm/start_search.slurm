#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=EvoStoneSearch
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=nikolaid_548
#SBATCH --output ./slurm/logs/slurm-%j-search.out
#SBATCH --error ./slurm/logs/slurm-%j-search.out

echo "========== SLURM JOB INFO =========="
echo
echo "The job will be started on the following node(s):"
echo $SLURM_JOB_NODELIST
echo
echo "Slurm User:         $SLURM_JOB_USER"
echo "Run Directory:      $(pwd)"
echo "Job ID:             $SLURM_JOB_ID"
echo "Job Name:           $SLURM_JOB_NAME"
echo "Partition:          $SLURM_JOB_PARTITION"
echo "Number of nodes:    $SLURM_JOB_NUM_NODES"
echo "Number of tasks:    $SLURM_NTASKS"
echo "Submitted From:     $SLURM_SUBMIT_HOST"
echo "Submit directory:   $SLURM_SUBMIT_DIR"
echo "Hostname:           $(hostname)"
echo
echo "Dashboard Host:     |$(hostname):8787|"
echo

echo
echo "========== Start =========="
date

echo
echo "========== Setup =========="


echo
echo "========== Starting Singularity .NET script =========="
singularity exec --cleanenv singularity/ubuntu_dotnet dotnet bin/DeckSearch.dll config/paladin_surr_fcnn_me.tml

echo
echo "========== Done =========="
date
