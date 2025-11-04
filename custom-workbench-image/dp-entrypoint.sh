#!/bin/bash
set -e

# Define source and destination paths
SRC_DIR="/opt/app-root/tmp/odh-data-processing"
DEST_DIR="/opt/app-root/src/odh-data-processing"

# Copy the notebooks to the user's persistent home directory if they don't exist
# This ensures the notebooks are present on the first and subsequent launches
if [ ! -d "$DEST_DIR" ]; then
  echo "Notebooks not found in home directory. Copying from $SRC_DIR"
  cp -r $SRC_DIR /opt/app-root/src/
else
  echo "$DEST_DIR already exists. $SRC_DIR will not be copied"
fi

# Execute the original command to start the Jupyter server
# This will use the NOTEBOOK_ROOT_DIR env var you set in the Containerfile
# Source code for start-notebook.sh can be seen at:
# https://github.com/opendatahub-io/notebooks/blob/main/jupyter/minimal/ubi9-python-3.12/start-notebook.sh
exec /opt/app-root/bin/start-notebook.sh
