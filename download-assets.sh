#!/bin/bash

# Directory to save the files
DIR="assets"

# URLs of the files to download
URLS=(
    "https://huggingface.co/intfloat/e5-small-v2/resolve/main/vocab.txt"
)

# Create the directory if it doesn't exist
mkdir -p $DIR

# Loop through each URL
for URL in "${URLS[@]}"; do
    # Extract the file name from the URL
    FILE_NAME=$(basename $URL)
    
    # Full path to the file
    FILE_PATH="$DIR/$FILE_NAME"
    
    # Check if the file already exists
    if [ -f "$FILE_PATH" ]; then
        echo "File $FILE_NAME already exists."
    else
        # Download the file
        echo "Downloading $FILE_NAME..."
        curl -o "$FILE_PATH" "$URL"
    fi
done
