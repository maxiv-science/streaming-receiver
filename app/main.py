import yaml
import uvicorn
import asyncio
import argparse
import zmq.asyncio
from fastapi import FastAPI
from receiver.queuey import Queuey
from receiver.detector import Detector, Eiger, Lambda, PilatusPipeline
from receiver.collector import Collector
from receiver.filewriter import FileWriter
from receiver.forwarder import Forwarder

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str, help='yaml configuration file')
parser.add_argument('detector', type=str, help='Name of detector in config file')
args = parser.parse_args()

with open(args.config_file) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)[args.detector]
    
print(config)

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

app = FastAPI()

@app.on_event('startup')
async def startup_event():
    context = zmq.asyncio.Context()
    forwarder = Forwarder(context, 123456)
    asyncio.create_task(collector.run(worker_queue, writer_queue, [forwarder,]))

@app.get('/status')
async def root():
    return collector.status

@app.get('/received_frames')
async def root():
    return collector.received_frames

def main():
    uvicorn.run(app,
                host = '0.0.0.0')
    
if __name__ == '__main__':
    main()
    



