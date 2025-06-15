#!/bin/bash

# Get the first argument as the file path
configFile="$1"

# Check if the file exists
if [[ -f "$configFile" ]]; then
    :
else
    echo "File does not exist: $configFile"
    exit 1
fi

# Get number of CPU cores
numCores=$(nproc)

echo "Changing the number of workers to $numCores in $configFile"
# Replace the line exports.workers = 1; with exports.workers = <numCores>;
sed -i "s/^exports\.workers = [0-9]\+;/exports.workers = $numCores;/" "$configFile"
