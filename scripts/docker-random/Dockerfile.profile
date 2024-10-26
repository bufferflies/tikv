FROM amazonlinux:2023.5.20241001.1

RUN dnf update -y && dnf install -y \
    binutils \
    jemalloc-devel \
    graphviz \
    ghostscript
