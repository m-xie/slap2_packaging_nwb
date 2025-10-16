# hash:sha256:05e8f2d681708d4d5852d5a13971aeab231ba96e20829b7fad661a35c0e215fa
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

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.biosyntax \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev
