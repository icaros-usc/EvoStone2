NUM_WORKERS="$1"

for (( worker_id = 0; worker_id < $NUM_WORKERS; worker_id++ ))
do
sbatch slurm/start_worker.slurm ${worker_id}
done