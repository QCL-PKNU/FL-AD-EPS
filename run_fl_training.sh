#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Start the server
python FL_server.py --retrain &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 4`; do
    echo "Starting client $i"
    python FL_client.py --retrain --cid=${i} &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait