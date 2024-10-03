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


#@pytest.mark.skipif(
#    "not config.getoption('repub')",
#    reason="explicitly enable --repub",
#)
@pytest.mark.asyncio
async def test_repub(receiver_process, stream_stins, tmp_path) -> None:
    await receiver_process(
        {
            "class": "Detector",
            "dcu_host_purple": "127.0.0.1",
            "data_port": 5556,
            "dset_name": "/entry/instrument/zyla/data",
        }
    )

    await receiver_process(
        {
            "class": "Detector",
            "dcu_host_purple": "buffer",
            "dcu_port_purple": 8999,
            "data_port": 4446,
            "dset_name": "/entry/instrument/zyla/data",
        },
        5001,
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        stb = await session.get("http://localhost:5000/status")
        while st.status != 200 or stb.status != 200:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")
            stb = await session.get("http://localhost:5000/status")

    ntrig = 20

    filename = tmp_path / "test.h5"
    # bfilename = tmp_path / "buffer.h5"
    asyncio.create_task(custom_stins(9999, str(filename), ntrig, 13, 17))
    # asyncio.create_task(custom_stins(8999, str(bfilename), ntrig, 3,7))

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/received_frames")
        stb = await session.get("http://localhost:5001/received_frames")
        content = await st.json()
        contentb = await stb.json()
        logging.debug("frames is %s", content)
        while content["value"] < ntrig:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/received_frames")
            stb = await session.get("http://localhost:5001/received_frames")
            content = await st.json()
            contentb = await stb.json()
            logging.debug("frames is %s %s", content, contentb)

        st = await session.get("http://localhost:5000/status")
        content = await st.json()
        while content["state"] == "running":
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")
            content = await st.json()
            logging.debug("status is %s", content)

        assert content["state"] == "idle"

    with h5py.File(filename) as f:
        assert f["entry/instrument/zyla/data"].shape == (17 - 13, 2000, 4000)
        seq = f["entry/instrument/zyla/sequence_number"][:]
        assert list(seq) == list(range(13, 17))

    bfilename = tmp_path / "test_from_7.h5"
    with h5py.File(bfilename) as f:
        assert f["entry/instrument/zyla/data"].shape == (6, 2000, 4000)
        seq = f["entry/instrument/zyla/sequence_number"][:]
        assert list(seq) == list(range(7, 13))
