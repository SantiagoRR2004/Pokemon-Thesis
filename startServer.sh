#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Navigate to the pokemon-showdown directory
cd pokemon-showdown || { echo "Directory 'pokemon-showdown' not found!"; exit 1; }

# Start Pokemon Showdown with the specified options
node pokemon-showdown $SERVER_PORT start --no-security
