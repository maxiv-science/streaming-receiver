import asyncio
import logging

import aiohttp
import pytest
import zmq.asyncio


@pytest.mark.asyncio
async def test_simple(
    streaming_receiver,
    stream_stins,
) -> None:

    await streaming_receiver({"class":"Detector",
                              "dcu_host_purple": "127.0.0.1",
                              "data_port": 23006,
                              "dset_name": "/entry/instrument/zyla/data"})

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        while st.status != 200:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")

    context = zmq.asyncio.Context()
    ntrig = 10
    asyncio.create_task(stream_stins(context, 9999, ntrig))

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

        assert content["state"] == "idle"