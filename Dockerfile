FROM harbor.maxiv.lu.se/registry.gitlab.com/maxiv/docker/conda-build:latest AS build

ARG version

WORKDIR /tmp

COPY . /tmp

RUN conda build ./recipe

RUN conda create -n streaming --use-local streaming-receiver && \
    conda-pack -n streaming -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

FROM harbor.maxiv.lu.se/dockerhub/library/ubuntu:jammy AS runtime
ENV PATH /venv/bin:$PATH
COPY --from=build /venv /venv
RUN apt-get update && apt-get install -y build-essential
RUN pip3 install -U dectris-compression


