import zmq
import json
import numpy as np
import cbor2
from itertools import count
from threading import Thread
from bitshuffle import compress_lz4
from .queuey import Queuey
from .processing import convert_tot, decompress_cbf, unpack_mono12p

class PilatusPipeline():
    def __init__(self, config):
        self.compress = config.get('compress', True)
        self.rotation = config.get('rotate', False)
        self.tot = config.get('tot', None)
        if self.tot:
            self.tot_tensor = np.load(self.tot)['tot_to_energy_tensor']
            print('tot_tensor', self.tot_tensor.shape)
            
    def __call__(self, header, parts):
        img = np.empty(header['shape'], dtype=np.int32)
        decompress_cbf(parts[1], img)
        header['compression'] = 'none'
               
        if self.tot:
            output = np.empty(img.shape, dtype=np.float32)
            convert_tot(img, self.tot_tensor, output)
            img = output
            header['type'] = 'float32'
        
        # rotation for the custom cosaxs L shaped pilatus 2M
        if self.rotation:
            img = np.rot90(img, -1)
            img = np.ascontiguousarray(img)
            header['shape'] = header['shape'][::-1]
               
        if self.compress:
            header['compression'] = 'bslz4'
            img = compress_lz4(img)
            
        return [header, img]
    
class OrcaPipeline():
    def __init__(self, config):
        pass
    
    def __call__(self, header, parts):
        if header['type'] == 'mono12p':
            img = np.empty(header['shape'], dtype=np.uint16)
            unpack_mono12p(parts[1], len(parts[1]), img)
            header['type'] = 'uint16'
        else:
            img = parts[1]
        return [header, img]
        
        
class Detector():
    """Detectors that use the standard streaming format
    """
    def __init__(self, pipeline=None):
        self.context = zmq.Context(2)
        self.threads = []
        self.pipeline = pipeline
        
    def run(self, config, queue):
        nworkers = config.get('nworkers', 1)
        for i in range(nworkers):
            t = Thread(target=self.worker, args=(config, queue))
            t.start()
            self.threads.append(t)
        
    def worker(self, config, queue: Queuey):
        data_pull = self.context.socket(zmq.PULL)
        host = config['dcu_host_purple']
        port = config.get('dcu_port_purple', 9999)
        data_pull.connect(f'tcp://{host}:{port}')
        
        while True:
            parts = data_pull.recv_multipart(copy=False)
            header = json.loads(parts[0].bytes)
            #print(header)
            if header['htype'] == 'image':
                if self.pipeline:
                    output = self.pipeline(header, parts)
                else:
                    output = [header, *parts[1:]]
                queue.put(output)
            else:
                rest = [json.loads(p.bytes) for p in parts[1:]]
                queue.put([header, *rest]) 

class Eiger(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        self._msg_number = count(0)
    
    def handle_header(self, header, parts, queue):
        info = json.loads(parts[1].bytes)
        appendix = json.loads(parts[8].bytes)
        meta_header = {'htype': 'header',
                        'msg_number': next(self._msg_number),
                        'filename': appendix['filename']}
        keys = ['count_time', 
                'countrate_correction_applied',
                'countrate_correction_count_cutoff',
                'photon_energy',
                'threshold_energy',
                'flatfield_correction_applied',
                'virtual_pixel_correction_applied',
                'pixel_mask_applied',
                'nimages',
                'ntrigger',
                'trigger_mode']
                
        if header['header_detail'] != 'none':
            meta_info = {key: info[key] for key in keys}
        else:
            meta_info = {}
        queue.put([meta_header, meta_info])
        
    def handle_frame(self, header, parts, queue):
        info = json.loads(parts[1].bytes)
        compression = 'bslz4' if 'bs' in info['encoding'] else 'none'
        data_header = {'htype': 'image',
                        'msg_number': next(self._msg_number),
                        'frame': header['frame'],
                        'shape': info['shape'][::-1],
                        'type': info['type'],
                        'compression': compression}
        queue.put([data_header, parts[2]])
        
    def worker(self, config, queue: Queuey):
        data_pull = self.context.socket(zmq.PULL)
        host = config['dcu_host_purple']
        data_pull.connect(f'tcp://{host}:9999')
       
        while True:
            parts = data_pull.recv_multipart(copy=False)
            header = json.loads(parts[0].bytes)
            if header['htype'] == 'dimage-1.0':
                self.handle_frame(header, parts, queue)
            
            elif header['htype'] == 'dheader-1.0':
                self.handle_header(header, parts, queue)
                
            elif header['htype'] == 'dseries_end-1.0':
                end_header = {'htype': 'series_end',
                              'msg_number': next(self._msg_number)}
                queue.put([end_header,])
                
                
class Lambda(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        
    def worker(self, config, queue: Queuey):
        data_pull = []
        for i in range(4):
            sock = self.context.socket(zmq.PULL)
            host = config['dcu_host_purple'][i]
            port = 9010 + i
            sock.connect(f'tcp://{host}:{port}')
            data_pull.append(sock)
        
        last_meta_header = None
        
        while True:
            parts = []
            headers = []
            for s in data_pull:
                msgs = s.recv_multipart(copy=False)
                headers.append(json.loads(msgs[0].bytes))
                parts.append(msgs)
                
            for m in range(1, 4):
                if (headers[0]['htype'] != headers[m]['htype']) or \
                   (headers[0]['msg_number'] != headers[m]['msg_number']):
                    raise RuntimeError('Non matching lambda header messages', headers[0], headers[m])
            
            # add data blob of all the modules to the merged message
            if headers[0]['htype'] == "image":
                # add x, z, rotation and full shape to image header
                header = headers[0]
                header.update(last_meta_header)
                
                merged = [header,]
                for m in range(4):
                    merged.append(parts[m][1])
                queue.put(merged)
                
            elif headers[0]['htype'] == "header":
                meta = {'x': [], 
                        'y': [], 
                        'rotation': []} 
                x, y, rotation = [], [], []
                for m in range(4):
                    info = json.loads(parts[m][1].bytes)
                    for key in ['x', 'y', 'rotation']:
                        meta[key].append(int(info[key]))
                    meta['full_shape'] = info['full_shape']
                
                last_meta_header = meta
                queue.put([headers[0], meta])
                
            elif headers[0]['htype'] == "series_end":
                queue.put([headers[0],])


def decode_multi_dim_array(tag):
    shape, contents = tag.value
    # tuple of shape, dtype, blob
    return shape, *contents

def decode_compression(tag):
    algorithm, elem_size, encoded = tag.value
    return encoded

tag_decoders = {
    40: lambda tag: decode_multi_dim_array(tag),
    68: lambda tag: (np.uint8, tag.value),
    69: lambda tag: (np.uint16, tag.value),
    70: lambda tag: (np.uint32, tag.value),
    85: lambda tag: (np.float32, tag.value),
    56500: lambda tag: decode_compression(tag),
}

def tag_hook(decoder, tag):
    # print(tag.tag)
    tag_decoder = tag_decoders.get(tag.tag)
    return tag_decoder(tag) if tag_decoder else tag


class DectrisStream2(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        self._msg_number = count(0)

    def worker(self, config, queue: Queuey):
        data_pull = self.context.socket(zmq.PULL)
        host = config['dcu_host_purple']
        data_pull.connect(f'tcp://{host}:31001')

        while True:
            msg = data_pull.recv(copy=False)
            msg = cbor2.loads(msg.buffer, tag_hook=tag_hook)
            if msg['type'] == 'start':
                filename = json.loads(msg['user_data'])['filename']
                meta_header = {'htype': 'header',
                               'msg_number': next(self._msg_number),
                               'filename': filename}
                keys = ['count_time',
                        'countrate_correction_enabled',
                        'flatfield_enabled',
                        'virtual_pixel_interpolation_enabled',
                        'pixel_mask_enabled',
                        'number_of_images',
                        ]

                meta_info = {key: msg[key] for key in keys if key in msg}
                meta_info.update(msg['threshold_energy'])
                queue.put([meta_header,meta_info])

            elif msg['type'] == 'image':
                nthresh = len(msg['data'])
                compression = 'bslz4'  # if 'bs' in info['encoding'] else 'none'
                out = []
                for i in range(nthresh):
                    data = msg['data'][f"threshold_{i+1}"]
                    shape, dtype, blob = data

                    if not out:
                        dtype = dtype([]).dtype.name
                        data_header = {'htype': 'image',
                                       'msg_number': next(self._msg_number),
                                       'frame': msg['image_id'],
                                       'shape': shape,
                                       'type': dtype,
                                       'compression': compression}
                        if nthresh > 1:
                            data_header['shape'] = (nthresh, *shape)
                            data_header['chunks'] = (1, *shape)

                        out.append(data_header)

                    out.append(blob)

                queue.put(out)

            elif msg['type'] == 'end':
                end_header = {'htype': 'series_end',
                              'msg_number': next(self._msg_number)}
                queue.put([end_header, ])
                    
           
