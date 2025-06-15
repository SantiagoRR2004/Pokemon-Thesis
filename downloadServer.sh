#!/bin/bash

# Set the repository URL and directory
REPO_URL="https://github.com/smogon/pokemon-showdown.git"
DIR_NAME="pokemon-showdown"

# Function to get the latest version tag
get_latest_tag() {
    git tag -l 'v*' | sort -V | tail -n 1
}

# Function to update the repository
update_repo() {
    echo "Updating repository..."
    echo "Fetching latest tags..."
    git fetch --tags

    LATEST_TAG=$(get_latest_tag)
    echo "Latest tag found: $LATEST_TAG"

    echo "Checking out stable version..."
    git checkout "$LATEST_TAG"
    echo "Running npm install..."
    npm install
    if [ -f config/config-example.js ]; then
        echo "Copying config-example.js to config.js..."
        cp config/config-example.js config/config.js
    else
        echo "config-example.js not found. Skipping config copy."
    fi

    # Configure the server
    ../serverConfiguration.sh config/config.js
}

# Check if the directory exists
if [ -d "$DIR_NAME" ]; then
    echo "Directory '$DIR_NAME' exists. Checking for updates..."
    cd "$DIR_NAME" || exit
    update_repo
else
    echo "Directory '$DIR_NAME' does not exist. Cloning repository..."
    git clone "$REPO_URL"
    cd "$DIR_NAME" || exit
    update_repo
fi
