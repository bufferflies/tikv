# syntax=docker/dockerfile:1.3
# This Docker image contains a minimal build environment for TiKV
#
# It contains all the tools necessary to reproduce official production builds of TiKV

# We need to use CentOS 7 because many of our users choose this as their deploy machine.
# Since the glibc it uses (2.17) is from 2012 (https://sourceware.org/glibc/wiki/Glibc%20Timeline)
# it is our lowest common denominator in terms of distro support.

# Some commands in this script are structured in order to reduce the number of layers Docker
# generates. Unfortunately Docker is limited to only 125 layers:
# https://github.com/moby/moby/blob/a9507c6f76627fdc092edc542d5a7ef4a6df5eec/layer/layer.go#L50-L53

# We require epel packages, so enable the fedora EPEL repo then install dependencies.
# Install the system dependencies
# Attempt to clean and rebuild the cache to avoid 404s

# To avoid rebuilds we first install all Cargo dependencies


# The prepare image avoid ruining the cache of the builder
FROM amazonlinux:2022.0.20220504.1 as prepare
WORKDIR /tikv

RUN yum install -y findutils

# This step will always ruin the cache
# There isn't a way with docker to wildcard COPY and preserve the directory structure
COPY . .
RUN mkdir /output
RUN for component in $(find . -type f -name 'Cargo.toml' -exec dirname {} \; | sort -u); do \
  mkdir -p "/output/${component}/src" \
  && touch "/output/${component}/src/lib.rs" \
  && cp "${component}/Cargo.toml" "/output/${component}/Cargo.toml" \
  ; done


FROM amazonlinux:2022.0.20220504.1 as builder

RUN yum clean all && yum makecache

RUN yum install -y \
  make glibc-devel gcc patch \
  perl cmake3 && \
  yum clean all

ENV LIBRARY_PATH /usr/local/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Download proto zip
RUN ARCH=$(arch | sed s/aarch64/aarch_64/) && \
  curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-linux-${ARCH}.zip && \
  unzip -o protoc-3.14.0-linux-${ARCH}.zip -d ./proto 
RUN chmod 755 -R ./proto/bin
ENV BASE=/usr/local
# Copy into path
RUN cp ./proto/bin/protoc ${BASE}/bin
RUN cp -R ./proto/include/* ${BASE}/include

# Install Rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- --no-modify-path --default-toolchain none -y
SHELL ["/bin/bash", "-c"]
RUN echo $'[source.crates-io]\n\
replace-with = "aliyun"\n\
[source.aliyun]\n\
registry = "sparse+https://mirrors.aliyun.com/crates.io-index/"\n\
' > /root/.cargo/config.toml
ENV CARGO_UNSTABLE_SPARSE_REGISTRY true
ENV PATH /root/.cargo/bin/:$PATH

# Install the Rust toolchain
WORKDIR /tikv
COPY rust-toolchain.toml ./
RUN rustup self update \
  && rustup set profile minimal \
  && rustup default $(awk -F '"' '/channel/{print $2}' rust-toolchain.toml)

# For cargo
COPY scripts ./scripts
COPY etc ./etc
COPY Cargo.lock ./Cargo.lock

COPY --from=prepare /output/ ./

COPY Makefile ./

# Add full source code
COPY cmd/ ./cmd/
COPY components/ ./components/
COPY src/ ./src/

# Build binaries now
ARG GIT_FALLBACK="Unknown (no git or not git repo)"
ARG GIT_HASH=${GIT_FALLBACK}
ARG GIT_BRANCH=${GIT_FALLBACK}
ENV TIKV_BUILD_GIT_HASH=${GIT_HASH}
ENV TIKV_BUILD_GIT_BRANCH=${GIT_BRANCH}

# Use --mount=type=cache for the dependencies. Ref: https://github.com/moby/buildkit/blob/v0.10/frontend/dockerfile/docs/syntax.md#run---mounttypecache
RUN --mount=type=cache,target=/root/.cargo/registry \
  --mount=type=cache,target=/root/.cargo/git \
  --mount=type=cache,target=/tikv/target,sharing=locked \
  make release \
  && cp /tikv/target/release/tikv-server /tikv-server \
  && cp /tikv/target/release/cse-ctl /cse-ctl \
  && cp /tikv/target/release/tikv-worker /tikv-worker

# Export to a clean image
FROM amazonlinux:2022.0.20220504.1
COPY --from=builder /tikv-server /tikv-server
COPY --from=builder /cse-ctl /cse-ctl
COPY --from=builder /tikv-worker /tikv-worker

EXPOSE 20160 20180

ENTRYPOINT ["/tikv-server"]
