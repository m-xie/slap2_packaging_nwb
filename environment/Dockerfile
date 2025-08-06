# hash:sha256:0a642f42f199ed7ca80d59174a2fbedc720e0b33bb23984eb4b8b9de898316f5
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
    pynwb==3.1.1 \
    -e git+https://github.com/AllenInstitute/openscope_upload.git@8196a5b08f154fc4428b816e52ff521fe6951c81#egg=openscope-upload
