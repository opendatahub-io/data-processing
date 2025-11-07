# Custom Workbench for Data Processing

Open Data Hub has the ability for users to add and run [custom workbench images](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.24/html/managing_openshift_ai/creating-custom-workbench-images).

Below are guidelines on how to create a custom workbench image that is started up with the jupyter notebooks in this repository.

## Base Images

Depending on hardware resources (ex: GPU access) it is recommended to start with a jupyter-minimal image on python version 3.12.

Examples include:
```Containerfile
quay.io/modh/odh-workbench-jupyter-minimal-cuda-py312-ubi9
quay.io/modh/odh-workbench-jupyter-minimal-cpu-py312-ubi9
```

The following can be added in a `Containerfile`:

```Containerfile
FROM quay.io/modh/odh-workbench-jupyter-minimal-cuda-py312-ubi9
```

## Starting up Workbench with Example Notebooks at Runtime

Specific Jupyter notebooks can be the starting point of users during the start up of a custom workbench.

To configure this in a custom workbench users must download the notebooks when they are building the workbench image and set `NOTEBOOK_ROOT_DIR` to the path of the notebooks under `/opt/app-root/src`.

The variable `NOTEBOOK_ROOT_DIR` is used by [start-notebook.sh](https://github.com/opendatahub-io/notebooks/blob/main/jupyter/minimal/ubi9-python-3.12/start-notebook.sh) as the value of `--ServerApp.root_dir` as an argument to `jupyter lab`.

Since `/opt/app-root/src` is mounted by the notebook controller upon the start of a workbench image and any content in that directory [will be cleared](https://github.com/opendatahub-io/notebooks/tree/main/examples#opendatahub-dashboard) a script is required to move the notebooks.

An example script that accomplishes this is in `dp-entrypoint.sh` and below:

```bash
#!/bin/bash
set -e

# Define source and destination paths
SRC_DIR="/opt/app-root/tmp/data-processing"
DEST_DIR="/opt/app-root/src/data-processing"

# Copy the notebooks to the user's persistent home directory if they don't exist
# This ensures the notebooks are present on the first and subsequent launch
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
```

This script is used within this snippet of a `Containerfile` to set a repository of notebooks at startup.

```Containerfile
USER 1001

# Clone the repository to a temporary, non-mounted directory
RUN git clone https://github.com/opendatahub-io/data-processing.git -b stable-3.0 /opt/app-root/tmp/data-processing

# Copy a custom entrypoint script into the container
COPY --chown=1001:1 dp-entrypoint.sh /opt/app-root/bin/dp-entrypoint.sh

# Check to make sure entry point script can be executed by random high number user
RUN ls -l /opt/app-root/bin/dp-entrypoint.sh

# Set the NOTEBOOK_ROOT_DIR to the final destination in the user's home directory
ENV NOTEBOOK_ROOT_DIR="/opt/app-root/src/data-processing/notebooks/use-cases"

# Set the custom script as the new entrypoint
ENTRYPOINT ["/opt/app-root/bin/dp-entrypoint.sh"]
```

## Building Tesseract OCR Dependencies

One of the OCR engines available to use in Docling is [Tesseract OCR](https://docling-project.github.io/docling/examples/tesseract_lang_detection/).

To use tesseract OCR within Docling, tesseract OC and some of it's dependencies need to be downloaded, built from source, and linked in the custom workbench container before Docling is installed. Below is part of a `Containerfile` that does this.

```Containerfile
USER root

# Enable EPEL and install build dependencies
RUN rpm -Uvh --nosignature https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm || true && \
    dnf -y install \
      git gcc gcc-c++ make autoconf automake libtool pkgconfig \
      zlib-devel libjpeg-turbo-devel libpng-devel libtiff-devel libwebp-devel openjpeg2-devel

# Build and install Leptonica from source
ENV LEPTONICA_VERSION=1.83.1
RUN curl -L -o /tmp/leptonica.tar.gz https://github.com/DanBloomberg/leptonica/archive/refs/tags/${LEPTONICA_VERSION}.tar.gz && \
    tar -xzf /tmp/leptonica.tar.gz -C /tmp && \
    cd /tmp/leptonica-${LEPTONICA_VERSION} && \
    ./autogen.sh && \
    ./configure && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /tmp/leptonica*

# Ensure build and runtime can find Leptonica from /usr/local
ENV PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV LIBLEPT_HEADERSDIR="/usr/local/include"

# Build and install Tesseract from source
ENV TESSERACT_VERSION=5.4.1
RUN git clone --depth=1 --branch ${TESSERACT_VERSION} https://github.com/tesseract-ocr/tesseract /tmp/tesseract && \
    cd /tmp/tesseract && \
    ./autogen.sh && \
    ./configure --disable-static && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /tmp/tesseract

# Install language data
ENV TESSDATA_VERSION=4.1.0
ENV TESSDATA_PREFIX="/usr/share/tesseract/tessdata"
RUN mkdir -p ${TESSDATA_PREFIX} && \
    git clone --depth=1 --branch ${TESSDATA_VERSION} https://github.com/tesseract-ocr/tessdata ${TESSDATA_PREFIX}

# Build and install tesserocr from source (uses pkg-config to find tesseract/leptonica)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-binary=:all: tesserocr
```

## Building and Pushing the Image

An example `Containerfile`, and shell script (`dp-entrypoint.sh`) that copies notebooks at run time are included in this directory.

The shell script must have execute permission set at the user and group level before it is copied into the container at build time so that it can be executed by the container runtime.

If tesseract OCR and it's dependencies are being built using `podman`, the podman virtual machine should have 4 CPUs and 4096 MB of RAM.

Below are commands that can be run from this directory to build a custom workbench container image.

```
chmod ug+x dp-entrypoint.sh
podman machine stop
podman machine set --cpus 4 --memory 4096
podman machine start
podman build -t custom-dp-wb-image:latest .
podman push localhost/custom-dp-wb-image:latest <CONTAINER REGISTRY>
```

## Workbench Size Recommendations

When users select the container size for the Workbench in the Workbench GUI, they are advised to at least use the `Medium` size container.
This will have 3-6 CPUs and at least 24 GB of memory for the underlying container.

## Consuming Data from S3

To consume data from S3 in your workbench please refer to this [tutorial](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/pdf/working_with_data_in_an_s3-compatible_object_store/Red_Hat_OpenShift_AI_Cloud_Service-1-Working_with_data_in_an_S3-compatible_object_store-en-US.pdf) on how to include files from S3 in a Jupyter notebook.
