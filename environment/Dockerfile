# hash:sha256:444454a71cac42c12061da511141752b2ee3feec4e86f981f29ff86b7365128f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    argparse \
    h5py==3.14.0 \
    harp-python==0.4.1 \
    hdmf-zarr==0.11.2 \
    pynwb==3.1.1 \
    -e git+https://github.com/AllenInstitute/openscope_upload.git@d54ae99620cb81f92cffa75ecf2fe66e9ee2e3b2#egg=openscope-upload
