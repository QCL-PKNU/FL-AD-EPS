#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Start the server
python preprocess.py eps1
sleep 0.1
python centralized.py --retrain --name "USAD_EPS1"

python preprocess.py eps2
sleep 0.1
python centralized.py --retrain --name "USAD_EPS2"

python preprocess.py eps3
sleep 0.1
python centralized.py --retrain --name "USAD_EPS3"

python preprocess.py eps4
sleep 0.1
python centralized.py --retrain --name "USAD_EPS4"


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait