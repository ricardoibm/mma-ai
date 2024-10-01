FROM registry.access.redhat.com/ubi9/ubi as builder

RUN dnf update -y && dnf -y groupinstall 'Development Tools' && dnf install -y \
  cmake git ninja-build \
  && dnf clean all


RUN git clone --recursive https://github.com/DanielCasali/OpenBLAS.git && cd OpenBLAS && \
    make -j$(nproc --all) TARGET=POWER10 USE_THREAD=1 && \
    make PREFIX=/opt/OpenBLAS install && \
    cd /

RUN git clone https://github.com/DanielCasali/llama.cpp.git && cd llama.cpp && sed -i "s/powerpc64le/native -mvsx -mtune=native -D__POWER10_VECTOR__/g" ggml/src/CMakeLists.txt && \
    mkdir build; \
    cd build; \
    cmake -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLAS_INCLUDE_DIRS=/opt/OpenBLAS/include -G Ninja ..; \
    cmake --build . --config Release

CMD bash

FROM registry.access.redhat.com/ubi9/ubi 

COPY --from=builder --chmod=755 /llama.cpp/build/bin/llama-server /usr/local/bin
COPY --from=builder --chmod=644 /llama.cpp/build/src/libllama.so /llama.cpp/build/src/libllama.so
COPY --from=builder --chmod=644 /llama.cpp/build/ggml/src/libggml.so /llama.cpp/build/ggml/src/libggml.so


ENTRYPOINT [ "/usr/local/bin/llama-server", "--host", "0.0.0.0"]
