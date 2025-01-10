#!/bin/bash

# Run a command in the background (e.g., sleep for 100 seconds)
python3 ddim.py &

# Capture the PID of the background process
PID=$!

# Print the PID
echo "The PID of the background process is: $PID"

# Perform some action (e.g., wait for the process to complete)
wait $PID

# Print a message after the process completes
echo "The background process with PID $PID has completed."
