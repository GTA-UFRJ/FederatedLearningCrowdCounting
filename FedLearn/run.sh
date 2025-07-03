#!/bin/bash

echo "Starting server"

for j in $(seq 0 9); do
    for i in $(seq 0 4); do
        python server.py &
        sleep 3  # Sleep for 3s to give the server enough time to start
        echo "Starting client $i"
        python client_dsnet.py --partition-id $i --n-split $j --number-clients 5 --name dsnet_ucsd_5_100_1_lr7 --dataset ucsd &
    done
    wait
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
