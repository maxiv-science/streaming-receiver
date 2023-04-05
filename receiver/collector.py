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
            

# seqlock algorithm from linux to threadsafe read data while producer could be updating the data
# works by updating a counter and if the counter is even the data is good and if the counter is odd
# the data is being written to
class SeqLock():
    def __init__(self, data):
        self.counter = 0
        self.data = data
        
    def write(self, data):
        self.counter += 1
        self.data = data
        self.counter += 1
        
    def _try_read(self):
        counter = self.counter
        # write in progress if counter is odd
        if (counter & 1 != 0):
            return False, None
        
        data = self.data.copy()
        # make sure counter was not modified while copying data
        return counter == self.counter, data
        
    def read(self):
        success, data = self._try_read()
        while not success:
            success, data = self._try_read()
        return data
        
            
class Collector():
    def __init__(self):
        self.frame_rate = 0
        self.downsample = 1
        self.received_frames = 0
        self.last_frame = SeqLock([{}, b''])
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
                self.last_frame.write(parts)
            writer_queue.put(parts)
            for forwarder in forwarders:
                await forwarder.forward(parts)
