# hash:sha256:dfb445c698ebb26fa8c1ddce79aad2f3f0dfe258ab9e74c324084681cc171fd8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-nwb-utils==0.2.5 \
    argparse \
    h5py==3.14.0 \
    harp-python==0.4.1 \
    hdmf-zarr==0.11.2 \
    matplotlib==3.9.4 \
    pynwb==3.1.1 \
    pyyaml==6.0.3 \
    -e git+https://github.com/AllenInstitute/openscope_upload.git@d54ae99620cb81f92cffa75ecf2fe66e9ee2e3b2#egg=openscope-upload
