#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Start the server
python preprocess.py eps1
sleep 0.1
python centralized.py --test --name "ML1"

python preprocess.py eps2
sleep 0.1
python centralized.py --test --name "ML2"

python preprocess.py eps3
sleep 0.1
python centralized.py --test --name "ML3"

python preprocess.py eps4 
sleep 0.1
python centralized.py --test --name "ML4"


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait