FROM harbor.maxiv.lu.se/daq/conda-build:latest AS build

ARG version

RUN conda create -n streaming streaming-receiver==${version} && \
    conda-pack -n streaming -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

FROM harbor.maxiv.lu.se/dockerhub/library/ubuntu:jammy AS runtime
ENV PATH /venv/bin:$PATH
COPY --from=build /venv /venv



