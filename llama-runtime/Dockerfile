FROM ubuntu:22.04 as builder

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake git ninja-build \
  && rm -rf /var/lib/apt/lists/*


RUN git clone --recursive https://github.com/DanielCasali/OpenBLAS.git && cd OpenBLAS && \
    make -j$(nproc --all) TARGET=POWER10 DYNAMIC_ARCH=1 && \
    make PREFIX=/opt/OpenBLAS install && \
    cd /

RUN git clone https://github.com/DanielCasali/llama.cpp.git && cd llama.cpp && sed -i "s/powerpc64le/native -mvsx -mtune=native -D__POWER10_VECTOR__/g" ggml/src/CMakeLists.txt && \
    mkdir build; \
    cd build; \
    cmake -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLAS_INCLUDE_DIRS=/opt/OpenBLAS/include -G Ninja ..; \
    cmake --build . --config Release

CMD bash

FROM ubuntu:22.04

COPY --from=builder --chmod=755 /llama.cpp/build/bin/llama-server /usr/local/bin
COPY --from=builder --chmod=644 /llama.cpp/build/src/libllama.so /llama.cpp/build/src/libllama.so
COPY --from=builder --chmod=644 /llama.cpp/build/ggml/src/libggml.so /llama.cpp/build/ggml/src/libggml.so
COPY --from=builder --chmod=644 /lib/powerpc64le-linux-gnu/libgomp.so.1 /lib/powerpc64le-linux-gnu/libgomp.so.1


ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0"]
