import asyncio
import json
import logging
import multiprocessing
import os
import random
from typing import AsyncIterator, Callable, Coroutine, Any

import aiohttp
import cbor2
import numpy as np
import pytest_asyncio
import uvicorn
import zmq
from aiohttp import ClientConnectionError
from pydantic_core import Url

import streaming_receiver.app.main as receiver_app
from stream1 import AcquisitionSocket

from pytest import Parser
import zipfile

from pytest_cov.embed import cleanup


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--repub",
        action="store_true",
        dest="repub",
        default=False,
        help="enable repub tests",
    )


class UvicornServer(multiprocessing.Process):
    def __init__(self, config: uvicorn.Config):
        super().__init__()
        self.server = uvicorn.Server(config=config)
        self.config = config
        logging.info("started server with config %s", config)

    def stop(self) -> None:
        cleanup()
        self.terminate()
        self.join()

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.server.run()


@pytest_asyncio.fixture()
async def receiver_process() -> (
    AsyncIterator[Callable[[dict], Coroutine[None, None, None]]]
):
    server_tasks = []

    async def start_generator(conf: dict, port: int = 5000) -> None:
        a = receiver_app.app
        a.config = conf
        config = uvicorn.Config(a, port=port, log_level="debug")
        instance = UvicornServer(config=config)
        instance.start()

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    st = await session.get(f"http://localhost:{port}/status")
                    if st.status == 200:
                        break
                    else:
                        await asyncio.sleep(0.5)
                except ClientConnectionError:
                    await asyncio.sleep(0.5)

        server_tasks.append(instance)

    yield start_generator

    for server in server_tasks:
        server.stop()

    await asyncio.sleep(0.1)


@pytest_asyncio.fixture()
async def streaming_receiver() -> AsyncIterator[
    Callable[[dict], Coroutine[None, None, None]]
]:
    server_tasks = []

    async def start_receiver(conf: dict) -> None:
        os.environ["DETECTOR_CONFIG"] = json.dumps(conf)
        config = uvicorn.Config(receiver_app.app, port=5000, log_level="debug")
        server = uvicorn.Server(config)
        server_tasks.append((server, asyncio.create_task(server.serve())))
        while server.started is False:
            await asyncio.sleep(0.1)

    yield start_receiver

    for server, task in server_tasks:
        server.should_exit = True
        await task

    del os.environ["DETECTOR_CONFIG"]
    await asyncio.sleep(0.1)


@pytest_asyncio.fixture
async def stream_stins() -> (
    Callable[[zmq.Context[Any], int, int], Coroutine[Any, Any, None]]
):
    async def _make_stins(
        ctx: zmq.Context[Any], filename: str, port: int, nframes: int, meta: Any = None
    ) -> None:
        socket = AcquisitionSocket(ctx, Url(f"tcp://*:{port}"))
        acq = await socket.start(filename=filename, meta=meta)
        width = 2000
        height = 4000
        for frameno in range(nframes):
            img = np.zeros((width, height), dtype=np.uint16)
            for _ in range(20):
                img[random.randint(0, width - 1)][
                    random.randint(0, height - 1)
                ] = random.randint(0, 10)
            await acq.image(img, img.shape, frameno)
            await asyncio.sleep(0.1)
        await acq.close()
        await socket.close()

    return _make_stins


@pytest_asyncio.fixture
async def stream_eiger_dump() -> Callable[
    [
        zmq.Context[Any],
        os.PathLike[Any] | str,
        str,
        int,
        float,
        int,
    ],
    Coroutine[Any, Any, None],
]:
    async def _make_dump(
        ctx: zmq.Context[Any],
        filename: os.PathLike[Any] | str,
        dst_file: str,
        port: int,
        frame_time: float = 0.1,
        typ: int = zmq.PUSH,
    ) -> None:
        socket: zmq.Socket[Any] = ctx.socket(typ)
        socket.bind(f"tcp://*:{port}")

        with zipfile.ZipFile(filename) as zf:
            for file in zf.namelist():
                if file.endswith(".cbors"):  # optional filtering by filetype
                    with zf.open(file) as f:
                        while True:
                            try:
                                dump = cbor2.load(f)
                                frames = list(dump.value[1].values())[0].value[1]
                                logging.info("send frames %s", frames[0])
                                hdr = json.loads(frames[0])
                                if hdr["htype"] == "dheader-1.0":
                                    appendix = json.loads(frames[8])
                                    logging.info("appendix is %s", appendix)
                                    appendix["filename"] = dst_file
                                    frames[8] = json.dumps(appendix).encode("utf8")
                                await socket.send_multipart(frames)
                                await asyncio.sleep(frame_time)
                            except EOFError:
                                logging.warning("end of file reached")
                                break

                    break

        socket.close()

    return _make_dump
