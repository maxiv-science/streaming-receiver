import yaml
import asyncio
import argparse
import zmq.asyncio
from receiver.queuey import Queuey
from receiver.detector import Detector, Eiger, Lambda, PilatusPipeline
from receiver.collector import Collector
from receiver.filewriter import FileWriter
from receiver.forwarder import Forwarder

async def server(context, collector, config):
    port = config['rep_port']
    rep_socket = context.socket(zmq.REP)
    # tcp keepalive messages to make sure connection stays alive
    rep_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
    rep_socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 10)
    rep_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
    rep_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 1)
    rep_socket.bind(f'tcp://*:{port}')
    while True:
        req = await rep_socket.recv()
        if req == b'received_frames':
            await rep_socket.send_string(str(collector.received_frames))
        elif req == b'status':
            await rep_socket.send_json(collector.status)
        elif req == b'last_frame':
            await rep_socket.send_json(collector.last_frame[0], flags=zmq.SNDMORE)
            await rep_socket.send_multipart(collector.last_frame[1:])
    

async def main(config):
    class_name = config['class']
    if class_name == 'Eiger':
        detector = Eiger()
    elif class_name == 'Lambda':
        detector = Lambda()
    else:
        pipeline_name = config.get('pipeline', None)
        pipeline = None
        if pipeline_name == 'PilatusPipeline':
            pipeline = PilatusPipeline(config)
        detector = Detector(pipeline)

    worker_queue = Queuey()
    writer_queue = Queuey()
    detector.run(config, worker_queue)
    collector = Collector()
    writer = FileWriter(config['dset_name'])
    writer.run(worker_queue, writer_queue)
    
    context = zmq.asyncio.Context()
    forwarder = Forwarder(context, config['data_port'])
    await asyncio.gather(collector.run(worker_queue, writer_queue, [forwarder,]),
                         server(context, collector, config))
    
#if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str, help='yaml configuration file')
parser.add_argument('detector', type=str, help='Name of detector in config file')
args = parser.parse_args()

with open(args.config_file) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)[args.detector]
    
print(config)
asyncio.run(main(config))
