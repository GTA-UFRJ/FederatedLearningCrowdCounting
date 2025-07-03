#!/bin/bash

echo "Starting server"


for j in $(seq 0 4); do
    for i in $(seq 0 2); do
        python server.py --port 0.0.0.0:8008  &
        sleep 3  # Sleep for 3s to give the server enough time to start
        echo "Starting client $i"
        python non_client_dsnet.py --partition-id $i --n-split $j --number-clients 2 --name dsnet_rio_mall70_drone70_3_100_1 --port 127.0.0.1:8008 &
    done
    wait
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
