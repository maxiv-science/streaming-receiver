from contextlib import asynccontextmanager
from typing import AsyncGenerator

import yaml
import pickle
import uvicorn
import asyncio
import argparse
import zmq.asyncio
import logging
import os
from fastapi import FastAPI, Response, Request
from streaming_receiver.receiver.queuey import Queuey
from streaming_receiver.receiver.collector import Collector
from streaming_receiver.receiver.forwarder import Forwarder
from streaming_receiver.receiver.filewriter import FileWriter
from streaming_receiver.receiver.processing import downsample
from streaming_receiver.receiver.detector import Detector
from streaming_receiver.receiver import detector as available_classes
from streaming_receiver.receiver.utils import cancel_and_wait, done_callback

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    config = app.config

    class_name = config["class"]
    available_detectors = {
        k: v
        for k, v in available_classes.__dict__.items()
        if isinstance(v, type) and issubclass(v, Detector)
    }
    if class_name in available_detectors:
        detector_class = available_detectors[class_name]
    else:
        raise RuntimeError(f"Unknow detector name: {class_name}")
    pipeline_name = config.get("pipeline", None)
    available_pipelines = {
        k: v
        for k, v in available_classes.__dict__.items()
        if isinstance(v, type) and not issubclass(v, Detector)
    }
    if pipeline_name is None:
        pipeline = None
    elif pipeline_name in available_pipelines:
        pipeline = available_pipelines[pipeline_name](config)
    else:
        raise RuntimeError(f"Unknow pipeline name: {pipeline_name}")
    detector = detector_class(pipeline)

    worker_queue = Queuey()
    writer_queue = Queuey()
    detector.run(config, worker_queue)
    collector = Collector()
    writer = FileWriter(config["dset_name"])
    writer.run(worker_queue, writer_queue)

    context = zmq.asyncio.Context()
    forwarder = Forwarder(context, config["data_port"])
    collector_task = asyncio.create_task(
        collector.run(
            worker_queue,
            writer_queue,
            [
                forwarder,
            ],
        )
    )
    collector_task.add_done_callback(done_callback)

    app.state.collector = collector
    print("all done, start app")

    yield

    await forwarder.close()

    await collector.close()
    await cancel_and_wait(collector_task)


app = FastAPI(lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="yaml configuration file")
    parser.add_argument("detector", type=str, help="Name of detector in config file")
    args = parser.parse_args()

    with open(args.config_file) as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)[args.detector]

    logger.info("load config: %s", config)
    app.config = config
    port = config.get("api_port", 5000)
    api_loglevel = os.getenv("LOG_LEVEL", "INFO").lower()
    if api_loglevel == "info":
        api_loglevel = "warning"
    if api_loglevel == "debug":
        api_loglevel = "info"
    try:
        config = uvicorn.Config(app, port=port, host="0.0.0.0", log_level=api_loglevel)
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        print("exiting")
    # asyncio.run(main(config))


if __name__ == "__main__":
    main()


@app.get("/status")
def status(req: Request):
    return req.app.state.collector.status


@app.get("/received_frames")
async def received_frames(req: Request):
    return {"value": req.app.state.collector.received_frames}


@app.get("/downsample")
async def get_downsample(req: Request):
    return {"value": req.app.state.collector.downsample}


@app.post("/downsample")
async def post_downsample(req: Request, value: int):
    req.app.state.collector.downsample = value
    return {"value": value}


# since this is a sync function fastapi will run this function in a threadpool
# so make sure it is threadsafe
@app.get("/last_frame", response_class=Response)
def last_frame(req: Request):
    downsample_factor = req.app.state.collector.downsample
    last_frame = req.app.state.collector.last_frame.read()
    header = last_frame[0]
    if (
        downsample_factor > 1
        and header["compression"] == "none"
        and header["type"] == "uint16"
    ):
        img = downsample(last_frame[1], header["shape"], downsample_factor)
        header = header.copy()
        header["shape"] = img.shape
        payload = pickle.dumps([header, img])
    else:
        parts = []
        for p in last_frame:
            if isinstance(p, zmq.Frame):
                parts.append(p.bytes)
            else:
                parts.append(p)
        payload = pickle.dumps(parts)
    return Response(payload, media_type="image/pickle")
