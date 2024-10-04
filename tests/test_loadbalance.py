import asyncio
import logging

import aiohttp
import h5py
import pytest
import zmq.asyncio


@pytest.mark.asyncio
async def test_lb(streaming_receiver, stream_stins, tmp_path) -> None:
    await streaming_receiver(
        {
            "class": "Detector",
            "dcu_host_purple": ["127.0.0.1", "127.0.0.2"],
            "data_port": 23023,
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

    with h5py.File(filename) as f:
        assert f["entry/instrument/zyla/data"].shape == (ntrig, 2000, 4000)
        seq = f["entry/instrument/zyla/sequence_number"][:]
        assert list(seq) == list(range(ntrig))
