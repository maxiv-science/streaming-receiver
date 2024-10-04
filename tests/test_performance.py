import asyncio
import logging

import aiohttp
import h5py
import pytest
import zmq.asyncio


@pytest.mark.parametrize(
    "size, ntrig, delay, save",
    [(50, 50000, 0.00001, False), (4000, 1000, 0.005, False)],
)
@pytest.mark.asyncio
async def test_perf(
    streaming_receiver, stream_stins, tmp_path, size, ntrig, delay, save
) -> None:
    await streaming_receiver(
        {
            "class": "Detector",
            "dcu_host_purple": "127.0.0.1",
            "data_port": 23014,
            "nworkers": 1,
            "dset_name": "/entry/instrument/zyla/data",
        }
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        while st.status != 200:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")

    context = zmq.asyncio.Context()

    filename = ""
    if save:
        filename = tmp_path / "test.h5"

    asyncio.create_task(
        stream_stins(
            context, str(filename), 9999, ntrig, delay, width=size, height=size
        )
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/received_frames")
        content = await st.json()
        logging.debug("frames is %s", content)
        while content["value"] < ntrig:
            await asyncio.sleep(1)
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

    if save:
        with h5py.File(filename) as f:
            assert f["entry/instrument/zyla/data"].shape == (ntrig, size, size)
            seq = f["entry/instrument/zyla/sequence_number"][:]
            assert list(seq) == list(range(ntrig))
