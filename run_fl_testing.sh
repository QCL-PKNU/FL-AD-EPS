#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Start the server
python preprocess.py eps1
sleep 0.1
python centralized.py --test --use_fl --name "FL1"

python preprocess.py eps2
sleep 0.1
python centralized.py --test --use_fl --name "FL2"

python preprocess.py eps3
sleep 0.1
python centralized.py --test --use_fl --name "FL3"

python preprocess.py eps4
sleep 0.1
python centralized.py --test --use_fl --name "FL4"

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait