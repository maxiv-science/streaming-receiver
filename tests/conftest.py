import asyncio
import logging
import multiprocessing
import random
from typing import AsyncIterator, Callable, Coroutine, Any

import aiohttp
import numpy as np
import pytest_asyncio
import uvicorn
import zmq
from aiohttp import ClientConnectionError
from pydantic_core import Url

import app.main
from stream1 import AcquisitionSocket

from pytest import Parser


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
        self.terminate()

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.server.run()


@pytest_asyncio.fixture()
async def receiver_process() -> (
    AsyncIterator[Callable[[dict], Coroutine[None, None, None]]]
):
    server_tasks = []

    async def start_generator(conf: dict, port: int = 5000) -> None:
        a = app.main.app
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
