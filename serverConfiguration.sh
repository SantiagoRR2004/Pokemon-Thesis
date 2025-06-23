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

echo "Changing the server to not report battles"
# Replace the line exports.reportbattles = true; with exports.reportbattles = false;
sed -i "s/^exports\.reportbattles = true;/exports.reportbattles = false;/" "$configFile"

echo "Turning off the punishment monitor"
# Replace exports.monitorminpunishments = *; with exports.monitorminpunishments = 0;
sed -i "s/^exports\.monitorminpunishments = [0-9]\+;/exports.monitorminpunishments = 0;/" "$configFile"

echo "Disabling throttling"
# Replace exports.nothrottle = false; with exports.nothrottle = true;
sed -i "s/^exports\.nothrottle = false;/exports.nothrottle = true;/" "$configFile"

echo "Disabling IP checks"
# Replace exports.noipchecks = false; with exports.noipchecks = true;
sed -i "s/^exports\.noipchecks = false;/exports.noipchecks = true;/" "$configFile"