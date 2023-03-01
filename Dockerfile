FROM docker.maxiv.lu.se/conda-build:latest AS build

RUN conda index package && \
    conda create -n streaming -c file://package/ streaming-receiver && \
    conda-pack -n streaming -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

FROM harbor.maxiv.lu.se/dockerhub/library/ubuntu:jammy AS runtime
ENV PATH /venv/bin:$PATH
COPY --from=build /venv /venv



