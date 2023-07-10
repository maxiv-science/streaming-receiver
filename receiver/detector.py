import zmq
import json
import numpy as np
import cbor2
import logging
from itertools import count
from threading import Thread
from bitshuffle import compress_lz4
from .queuey import Queuey
from .processing import convert_tot, decompress_cbf, unpack_mono12p

logger = logging.getLogger(__name__)
class PilatusPipeline():
    def __init__(self, config):
        self.compress = config.get('compress', True)
        self.rotation = config.get('rotate', False)
        self.mask = config.get('mask', [])
        self.tot = config.get('tot', None)
        if self.tot:
            self.tot_tensor = np.load(self.tot)['tot_to_energy_tensor']
            print('tot_tensor', self.tot_tensor.shape)
        logger.info("initialised PilatusPipeline with compress:%r rotate:%r mask:%s tot:%s", self.compress, self.rotation, self.mask, self.tot)
            
    def __call__(self, header, parts):
        img = np.empty(header['shape'], dtype=np.int32)
        decompress_cbf(parts[1], img)
        header['compression'] = 'none'
        logger.debug("process call with hdr %s", header)
               
        if self.tot:
            output = np.empty(img.shape, dtype=np.float32)
            convert_tot(img, self.tot_tensor, output)
            img = output
            header['type'] = 'float32'
            logger.debug("converted TOT")
        
        # rotation for the custom cosaxs L shaped pilatus 2M
        if self.rotation:
            img = np.rot90(img, -1)
            img = np.ascontiguousarray(img)
            header['shape'] = header['shape'][::-1]
            logger.debug("rotated image")

        # mask for the custom cosaxs L shaped pilatus 2M, crop after rotate
        if self.mask:
            img[self.mask[0][0]:self.mask[0][1], self.mask[1][0]:self.mask[1][1]] = -1
            logger.debug("masked rectangle")
               
        if self.compress:
            header['compression'] = 'bslz4'
            img = compress_lz4(img)
            logger.debug("lz4 compressed image")
            
        return [header, img]
    
class OrcaPipeline():
    def __init__(self, config):
        pass
        logger.info("initialised OrcaPipeline")
    
    def __call__(self, header, parts):
        if header['type'] == 'mono12p':
            img = np.empty(header['shape'], dtype=np.uint16)
            unpack_mono12p(parts[1], len(parts[1]), img)
            header['type'] = 'uint16'
            logger.debug("unpacked mono12p")
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
        logger.info("initialised detector with pipeline %s", self.pipeline)

    def run(self, config, queue):
        nworkers = config.get('nworkers', 1)
        for i in range(nworkers):
            t = Thread(target=self.worker, args=(config, queue))
            t.start()
            self.threads.append(t)
        logger.info("created %d worker threads", nworkers)

    def worker(self, config, queue: Queuey):
        data_pull = self.context.socket(zmq.PULL)
        host = config['dcu_host_purple']
        port = config.get('dcu_port_purple', 9999)
        data_pull.connect(f'tcp://{host}:{port}')
        logger.info("connected to tcp://%s:%d", host, port)
        
        while True:
            parts = data_pull.recv_multipart(copy=False)
            header = json.loads(parts[0].bytes)
            logger.debug("received frame with header %s", header)
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
        logger.info("initialised Eiger")
    
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
        logger.info("processed meta_header: %s and meta_info: %s", meta_header, meta_info)
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
        logger.debug("handled frame with header %s", data_header)
        queue.put([data_header, parts[2]])
        
    def worker(self, config, queue: Queuey):
        data_pull = self.context.socket(zmq.PULL)
        host = config['dcu_host_purple']
        data_pull.connect(f'tcp://{host}:9999')
        logger.info("connected to tcp://%s:9999", host)
       
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
                logger.info("series end")
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
            logger.info("connected to tcp://%s:%d", host, port)
        
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
                logger.debug("received image %s", header)
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

                logger.info("received header with headers %s and meta: %s", headers[0], meta)
                last_meta_header = meta
                queue.put([headers[0], meta])
                
            elif headers[0]['htype'] == "series_end":
                queue.put([headers[0],])
                logger.info("series end %s", headers[0])


def decode_multi_dim_array(tag):
    shape, contents = tag.value
    # tuple of shape, dtype, blob
    return shape, *contents

def decode_compression(tag):
    algorithm, elem_size, encoded = tag.value
    return encoded

tag_decoders = {
    40: lambda tag: decode_multi_dim_array(tag),
    64: lambda tag: (np.uint8, tag.value),
    68: lambda tag: (np.uint8, tag.value),
    69: lambda tag: (np.uint16, tag.value),
    70: lambda tag: (np.uint32, tag.value),
    85: lambda tag: (np.float32, tag.value),
    56500: lambda tag: decode_compression(tag),
}

def tag_hook(decoder, tag):
    #print(tag.tag)
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
        logger.info("connected to tcp://%s:31001", host)

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
                logger.info("received start, meta_header %s and meta_info %s", meta_header, meta_info)
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
                        logger.debug("processed image with header %s", data_header)

                    out.append(blob)

                queue.put(out)

            elif msg['type'] == 'end':
                end_header = {'htype': 'series_end',
                              'msg_number': next(self._msg_number)}
                queue.put([end_header, ])
                logger.info("series end %s", end_header)
                    
           
