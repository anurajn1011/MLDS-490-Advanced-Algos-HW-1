
#!/usr/bin/env bash

set -u

### Run the validation scripts for the Policy Gradient problems.

# Create a directory for logs
mkdir -p logs

scripts=(
    "src/q2/pong_baseline.py"
    "src/q2/validation_edge_cases/validation_cartpole.py"
    "src/q2/validation_edge_cases/validation_pong.py"
)

for script in "${scripts[@]}"; do
    name="$(basename "${script%.py}")"
    logfile="logs/${name}.log"

    echo "[$(date)] Starting $script"

    python -u "$script" > "$logfile" 2>&1
    status=$?

    echo "[$(date)] Finished $script with exit code $status"

    if [ "$status" -ne 0 ]; then
        echo "ERROR: $script failed. See $logfile"
    fi
done

echo "[$(date)] All scripts finished."
