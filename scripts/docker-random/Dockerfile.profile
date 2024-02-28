FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN sed -i s@/archive.ubuntu.com/@/apt.ksyun.cn/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/apt.ksyun.cn/@g /etc/apt/sources.list && \
    apt-get update -y && apt-get install -y \
    binutils \
    libjemalloc-dev \
    graphviz \
    ghostscript
