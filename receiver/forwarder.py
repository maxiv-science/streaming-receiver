import zmq
import json
import struct
import asyncio
            
class Forwarder():
    def __init__(self, context, port):
        self.nclients = 0
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.push_socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 10)
        self.push_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        self.push_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 1)
        self.push_socket.bind(f'tcp://*:{port}')
        self.monitor = self.push_socket.get_monitor_socket()
        asyncio.create_task(self.event_monitor())
        
    async def event_monitor(self):
        while True:
            parts = await self.monitor.recv_multipart()
            event, value = struct.unpack("=hi", parts[0])
            if event == zmq.EVENT_ACCEPTED:
                self.nclients += 1
                print('client connected')
            elif event == zmq.EVENT_DISCONNECTED:
                self.nclients -= 1
                print('client disconnected')
                
    async def forward(self, parts):
        if self.nclients > 0:
            msgs = []
            for p in parts:
                if isinstance(p, dict):
                    if "type" in p and type(p["type"]) == type:
                        p["type"] = p["type"](1).dtype.name
                    msgs.append(json.dumps(p).encode())
                else:
                    msgs.append(p)
                    
            await self.push_socket.send_multipart(msgs, copy=False)
