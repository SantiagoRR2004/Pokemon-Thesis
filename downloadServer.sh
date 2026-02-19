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


    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "none")

    if [ "$CURRENT_TAG" = "$LATEST_TAG" ]; then
        echo "Already on the latest version ($LATEST_TAG). Skipping checkout."
    else

        echo "Checking out stable version..."
        git checkout "$LATEST_TAG"

        if [ -f config/config-example.js ]; then
            echo "Copying config-example.js to config.js..."
            cp config/config-example.js config/config.js
        else
            echo "config-example.js not found. Skipping config copy."
        fi

        # Configure the server
        ../serverConfiguration.sh config/config.js

        # Build the server
        npm install pg@8
        node build

        : <<'END_COMMENT'
        Need to replace line 79 in lib/net.ts with:

        const protocol = new URL(this.uri).protocol;

        The old line uses 'url.parse', which is deprecated:
        const protocol = url.parse(this.uri).protocol;
END_COMMENT

        echo "Modifying lib/net.ts to use URL instead of url.parse..."
        sed -i 's/const protocol = url\.parse(this\.uri)\.protocol;/const protocol = new URL(this.uri).protocol;/g' lib/net.ts
    fi
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
