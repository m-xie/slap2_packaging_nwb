# hash:sha256:9f2b89f053bb16d73f523ba6898b60dc4c6404836c05dde0d4a2b2fb1e8e8022
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    argparse \
    h5py==3.14.0 \
    harp-python==0.4.1 \
    hdmf-zarr==0.11.2 \
    pynwb==3.1.1 \
    -e git+https://github.com/AllenInstitute/openscope_upload.git@8196a5b08f154fc4428b816e52ff521fe6951c81#egg=openscope-upload
