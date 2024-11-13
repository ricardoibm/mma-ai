ARG BASE_IMAGE=ubi9/ubi

# Mutli-stage build to keep final image small. Otherwise end up with
# curl and openssl installed
FROM --platform=$BUILDPLATFORM $BASE_IMAGE AS stage1
ARG VERSION=0.27.0
RUN dnf update -y && \
    dnf install -y git gcc-c++ \
    bzip2 \
    ca-certificates \
    ca-certificates \
    && dnf clean all
RUN git clone --recursive https://github.com/DanielCasali/OpenBLAS.git && cd OpenBLAS && \
    make -j$(nproc --all) TARGET=POWER10 USE_THREAD=1 && \
    make PREFIX=/opt/OpenBLAS install && \
    rm -rf OpenBLAS && \
    cd /
ARG TARGETARCH
RUN test "$TARGETARCH" = 'amd64' && export ARCH='64'; \
    test "$TARGETARCH" = 'arm64' && export ARCH='aarch64'; \
    test "$TARGETARCH" = 'ppc64le' && export ARCH='ppc64le'; \
    curl -L "https://micromamba.snakepit.net/api/micromamba/linux-${ARCH}/${VERSION}" | \
    tar -xj -C "/tmp" "bin/micromamba"

FROM $BASE_IMAGE as setup-mamba

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"


COPY --from=stage1 /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt
COPY --from=stage1 /tmp/bin/micromamba "$MAMBA_EXE"
COPY --from=stage1 /opt/OpenBLAS /opt/OpenBLAS

ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER

COPY --chmod=0755 _dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --chmod=0755 _dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh


WORKDIR /tmp

# Default command for "docker run"
CMD ["/bin/bash"]

FROM setup-mamba
ENV PYTHON_VERSION=3.10
ENV CHANNEL=rocketce
ENV OPENCE_VERSION=1.9.1
ENV LD_LIBRARY_PATH=/opt/OpenBLAS/lib
ENV PKG_CONFIG_PATH=/opt/OpenBLAS/lib/pkgconfig
RUN dnf update -y && dnf install -y cmake gcc-c++ gfortran libxcrypt-compat libxcrypt && dnf clean all
WORKDIR /work
RUN micromamba config append channels ${CHANNEL}/label/${CHANNEL}-${OPENCE_VERSION} &&     micromamba config append channels ${CHANNEL} &&     micromamba config append channels defaults &&     micromamba install -y -n base python=${PYTHON_VERSION} git "pyarrow>=12.0.0" "grpcio<=1.60.0,>=1.49.1" langchain "pytorch-cpu>=1.11.0" altair=4 streamlit transformers && micromamba clean --all --yes
RUN mkdir -p /opt/rh/gcc-toolset-11/root/usr && ln -s /usr/bin /opt/rh/gcc-toolset-11/root/usr/bin
RUN /opt/conda/bin/pip install --upgrade 'streamlit' pymilvus httpx asyncio pypdf httpx asyncio pypdf "sentence-transformers>=3.1.1" accelerate #'grpcio<=1.60.0,>=1.49.1' 'ujson>=2.0.0' 'pyarrow>=12.0.0' 'minio>=7.0.0' 'scipy' 
RUN /opt/conda/bin/pip cache purge
RUN dnf erase -y cmake gcc-c++ gfortran && dnf clean all
COPY models--sentence-transformers--all-MiniLM-L6-v2  /work/models--sentence-transformers--all-MiniLM-L6-v2
COPY streamlit.py /work/
USER 1001
EXPOSE 8501
CMD [ "/opt/conda/bin/streamlit" , "run" , "/work/streamlit.py" ]
