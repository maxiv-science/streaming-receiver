import asyncio
import logging
import pickle

import aiohttp
import h5py
import numpy as np
import pytest
import zmq.asyncio
from bitshuffle import decompress_lz4


@pytest.mark.asyncio
async def test_legacy(receiver_process, stream_eiger_dump, tmp_path) -> None:
    await receiver_process(
        {
            "class": "Eiger",
            "dcu_host_purple": "127.0.0.1",
            "data_port": 23008,
            "dset_name": "/entry/instrument/eiger/data",
        }
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/status")
        while st.status != 200:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/status")
            # st.close()

    context = zmq.asyncio.Context()

    filename = tmp_path / "test.h5"
    asyncio.create_task(
        stream_eiger_dump(context, "tests/eiger_legacy_dump.zip", str(filename), 9999)
    )

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/received_frames")
        content = await st.json()
        logging.debug("frames is %s", content)
        while content["value"] < 9:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/received_frames")
            content = await st.json()
            logging.debug("frames is %s", content)

        await asyncio.sleep(1)

        await stream_eiger_dump(context, "tests/eiger_legacy_dump.zip", "", 9999)

        fr = await session.get("http://localhost:5000/last_frame")
        parts = pickle.loads(await fr.read())
        header = parts[0]
        logging.info("header of last %s", header)
        img = decompress_lz4(parts[1], header["shape"], dtype=header["type"])
        logging.debug("data %s", img)

    ntrig = 9

    with h5py.File(filename) as f:
        assert f["entry/instrument/eiger/data"].shape == (ntrig, 1065, 1030)
        seq = f["entry/instrument/eiger/sequence_number"][:]
        assert list(seq) == list(range(ntrig))
        assert np.array_equal(img, f["entry/instrument/eiger/data"][-1])
        assert not np.array_equal(img, f["entry/instrument/eiger/data"][-2])
