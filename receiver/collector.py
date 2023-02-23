import time
import asyncio
from .queuey import Queuey

async def ordered_recv(queue: Queuey):
    cache = {}
    next_msg_number = -1
    while True:
        parts = await queue.get_async()
        header = parts[0]
        # just pass on status messages
        if header['htype'] == 'status':
            yield parts
            continue
        
        msg_number = header['msg_number']
        if header['htype'] == 'header':
            next_msg_number = msg_number
            # clear cache and remove older leftover messages if previous series didn't properly finish
            for key in list(cache.keys()):
                if key < msg_number:
                    del cache[key]
                    
        if msg_number == next_msg_number:
            yield parts
            next_msg_number += 1
            while next_msg_number in cache:
                entry = cache.pop(next_msg_number)
                yield entry
                next_msg_number += 1
        else:
            cache[msg_number] = parts
            
class Collector():
    def __init__(self):
        self.received_frames = 0
        self.frame_rate = 0
        self.last_frame = None
        self.status = {'state': 'idle'}
        
    async def _update_metrics(self):
        while True:
            start = time.time()
            old = self.received_frames
            await asyncio.sleep(1.0)
            end = time.time()
            #print((self.received_frames - old) / (end - start))
        
    async def run(self, worker_queue, writer_queue, forwarders):
        asyncio.create_task(self._update_metrics())
        
        async for parts in ordered_recv(worker_queue):
            #print('collector', parts[0])
            header = parts[0]
            
            if header['htype'] == 'status':
                header.pop('htype')
                self.status = header
                continue
            elif header['htype'] == 'header':
                self.received_frames = 0
            elif header['htype'] == 'image':
                self.received_frames += 1
            writer_queue.put(parts)
            for forwarder in forwarders:
                await forwarder.forward(parts)
