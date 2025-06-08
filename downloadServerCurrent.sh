#!/bin/bash

# Set the repository URL and directory
REPO_URL="https://github.com/smogon/pokemon-showdown.git"
DIR_NAME="pokemon-showdown"


# Function to update the repository
update_repo() {
    echo "Updating repository..."
    git pull
    echo "Running npm install..."
    npm install
    if [ -f config/config-example.js ]; then
        echo "Copying config-example.js to config.js..."
        cp config/config-example.js config/config.js
    else
        echo "config-example.js not found. Skipping config copy."
    fi
}

# Check if the directory exists
if [ -d "$DIR_NAME" ]; then
    echo "Directory '$DIR_NAME' exists. Checking for updates..."
    cd "$DIR_NAME" || exit
    # Check if there are any changes to pull
    git remote update
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u})
    BASE=$(git merge-base @ @{u})

    if [ "$LOCAL" = "$REMOTE" ]; then
        echo "Repository is up-to-date. No action required."
    elif [ "$LOCAL" = "$BASE" ]; then
        echo "Local repository is behind. Updating..."
        update_repo
    else
        echo "Repository is ahead of remote or diverged. Please resolve manually."
    fi
else
    echo "Directory '$DIR_NAME' does not exist. Cloning repository..."
    git clone "$REPO_URL"
    cd "$DIR_NAME" || exit
    echo "Running npm install..."
    npm install
    echo "Copying config-example.js to config.js..."
    cp config/config-example.js config/config.js
fi
