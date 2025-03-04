import asyncio
import logging
import random

import aiohttp
import h5py
import numpy as np
import pytest
import zmq.asyncio
from pydantic_core import Url

from stream1 import AcquisitionSocket


async def custom_stins(port, filename, totalframes, start, stop) -> None:
    ctx = zmq.asyncio.Context()
    socket = AcquisitionSocket(ctx, Url(f"tcp://*:{port}"))
    acq = await socket.start(filename=filename)
    width = 2000
    height = 4000
    for frameno in range(totalframes):
        img = np.zeros((width, height), dtype=np.uint16)
        for _ in range(20):
            img[random.randint(0, width - 1)][
                random.randint(0, height - 1)
            ] = random.randint(0, 10)
        extra = {}
        if frameno < start or frameno >= stop:
            extra["discard"] = True
        await acq.image(img, img.shape, frameno, extra_fields=extra)
        await asyncio.sleep(0.1)
    await acq.close()
    await socket.close()
    ctx.destroy()


@pytest.mark.asyncio
async def test_discard(streaming_receiver, stream_stins, tmp_path) -> None:
    await streaming_receiver(
        {
            "class": "Detector",
            "dcu_host_purple": "127.0.0.1",
            "data_port": 23007,
            "dset_name": "/entry/instrument/zyla/data",
        }
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        while st.status != 200:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")

    ntrig = 10

    filename = tmp_path / "test.h5"
    asyncio.create_task(custom_stins(9999, str(filename), ntrig, 3, 7))

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/received_frames")
        content = await st.json()
        logging.debug("frames is %s", content)
        while content["value"] < ntrig:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/received_frames")
            content = await st.json()
            logging.debug("frames is %s", content)

        st = await session.get("http://localhost:5000/status")
        content = await st.json()
        while content["state"] == "running":
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")
            content = await st.json()
            logging.debug("status is %s", content)

        assert content["state"] == "idle"

    with h5py.File(filename) as f:
        assert f["entry/instrument/zyla/data"].shape == (7 - 3, 2000, 4000)
        seq = f["entry/instrument/zyla/sequence_number"][:]
        assert list(seq) == list(range(3, 7))
