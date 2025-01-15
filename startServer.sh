#!/bin/bash

# Navigate to the pokemon-showdown directory
cd pokemon-showdown || { echo "Directory 'pokemon-showdown' not found!"; exit 1; }

# Start Pokemon Showdown with the specified options
node pokemon-showdown start --no-security
