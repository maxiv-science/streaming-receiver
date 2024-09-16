import asyncio
import random
from typing import AsyncIterator, Callable, Coroutine, Any

import numpy as np
import pytest_asyncio
import uvicorn
import zmq
from pydantic_core import Url

import app
from app.main import main
from stream1 import AcquisitionSocket


@pytest_asyncio.fixture()
async def streaming_receiver() -> AsyncIterator[Callable[[dict], Coroutine[None, None, None]]]:
    server_tasks = []

    async def start_receiver(conf: dict) -> None:
        a = app.main.app
        a.config = conf
        config = uvicorn.Config(
            a, port=5000, log_level="debug"
        )
        server = uvicorn.Server(config)
        server_tasks.append((server, asyncio.create_task(server.serve())))
        while server.started is False:
            await asyncio.sleep(0.1)

    yield start_receiver

    for server, task in server_tasks:
        server.should_exit = True
        await task

    await asyncio.sleep(0.1)

@pytest_asyncio.fixture
async def stream_stins() -> Callable[
    [zmq.Context[Any], int, int], Coroutine[Any, Any, None]
]:
    async def _make_stins(ctx: zmq.Context[Any], filename: str, port: int, nframes: int, meta: Any=None) -> None:
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