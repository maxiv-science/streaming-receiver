import asyncio
import logging
import random
from typing import Any

import aiohttp
import h5py
import numpy as np
import pytest
import zmq.asyncio
from pydantic_core import Url

from stream1 import AcquisitionSocket


async def mixed_stream(
    ctx: zmq.Context[Any], filename: str, port: int, nframes: int, meta: Any = None
) -> None:
    socket = AcquisitionSocket(ctx, Url(f"tcp://*:{port}"))
    acq = await socket.start(filename=filename, meta=meta)
    width = 2000
    height = 4000
    msg_numbers = [next(acq._msg_number) for _ in range(nframes)]
    multiparts = []
    for frameno in range(nframes):
        img = np.zeros((width, height), dtype=np.uint16)
        img[frameno, frameno] = 1

        header = {
            "htype": "image",
            "frame": frameno,
            "shape": img.shape,
            "type": str(img.dtype),
            "compression": "none",
            "msg_number": msg_numbers[frameno],
        }
        multiparts.append((header, img))

    random.shuffle(multiparts)

    for h, d in multiparts:
        await acq._socket.send_json(h, flags=zmq.SNDMORE)
        await acq._socket.send(d, copy=False)
        await asyncio.sleep(0.1)

    await acq.close()
    await socket.close()


@pytest.mark.asyncio
async def test_order(streaming_receiver, tmp_path) -> None:
    await streaming_receiver(
        {
            "class": "Detector",
            "dcu_host_purple": "127.0.0.1",
            "data_port": 23009,
            "dset_name": "/entry/instrument/zyla/data",
        }
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        while st.status != 200:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")

    context = zmq.asyncio.Context()

    ntrig = 10
    filename = tmp_path / "test.h5"
    await mixed_stream(context, str(filename), 9999, ntrig)

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        content = await st.json()
        while content["state"] == "running":
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")
            content = await st.json()
            logging.debug("status is %s", content)

        assert content["state"] == "idle"

    with h5py.File(filename) as f:
        assert f["entry/instrument/zyla/data"].shape == (ntrig, 2000, 4000)
        seq = f["entry/instrument/zyla/sequence_number"][:]
        assert list(seq) == list(range(ntrig))
        for i in range(ntrig):
            assert f["entry/instrument/zyla/data"][i][i, i] == 1
