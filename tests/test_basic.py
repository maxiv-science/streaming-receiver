import asyncio
import logging
import pickle

import aiohttp
import h5py
import numpy as np
import pytest
import zmq.asyncio


async def consume(num) -> None:
    c = zmq.asyncio.Context()
    s = c.socket(zmq.PULL)
    s.connect("tcp://localhost:23006")
    for i in range(num):
        logging.info("recv %d", i)
        task = s.poll()
        waiting = asyncio.gather(*[task])
        done = await waiting
        logging.info("done %s", done)
        data = s.recv_multipart(copy=False)
        logging.info("i %d, data %s", i, data)
    c.destroy()


@pytest.mark.asyncio
async def test_simple(receiver_process, stream_stins, tmp_path) -> None:

    await receiver_process(
        {
            "class": "Detector",
            "dcu_host_purple": "127.0.0.1",
            "data_port": 23006,
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
    ctask = asyncio.create_task(consume(ntrig))

    filename = tmp_path / "test.h5"
    asyncio.create_task(stream_stins(context, str(filename), 9999, ntrig))

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

        fr = await session.get("http://localhost:5000/last_frame")
        parts = pickle.loads(await fr.read())
        header = parts[0]
        img = np.frombuffer(parts[1], dtype=header["type"]).reshape(header["shape"])

        logging.debug("data %s", img)

    await ctask

    with h5py.File(filename) as f:
        assert f["entry/instrument/zyla/data"].shape == (ntrig, 2000, 4000)
        seq = f["entry/instrument/zyla/sequence_number"][:]
        assert list(seq) == list(range(ntrig))
        assert np.array_equal(img, f["entry/instrument/zyla/data"][-1])
        assert not np.array_equal(img, f["entry/instrument/zyla/data"][-2])
