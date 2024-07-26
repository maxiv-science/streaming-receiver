import logging
import os
import json
import time

import h5py
import bitshuffle
import numpy as np
import logging
from threading import Thread
from datetime import datetime
from .queuey import Queuey

logger = logging.getLogger(__name__)
def get_time_str():
    return datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S.%f')

class FileWriter():
    def __init__(self, dset_name):
        self._fh = None
        self._thread = None
        self._dset_name = dset_name
        self._number_dset_name = None
        logger.info("initialised FileWriter for dset %s", self._dset_name)
        
    def run(self, worker_queue: Queuey, writer_queue: Queuey):
        self._thread = Thread(target=self._main, args=(worker_queue, writer_queue))
        self._thread.start()
        
    def _main(self, worker_queue, writer_queue):
        while True:
            parts = writer_queue.get()
            header = parts[0]
            #print('writer', header)
            if header['htype'] == 'header':
                self._handle_start(header, parts, worker_queue)
            elif header['htype'] == 'series_end':
                self._handle_end(header, worker_queue)
            else:
                self._handle_frame(header, parts)

    def save_dict_to_h5(self, data, group):
        for key, value in data.items():
            if isinstance(value, dict):
                ng = group.create_group(key)
                self.save_dict_to_h5(value, ng)
            else:
                group.create_dataset(key, data=value)

    def _handle_start(self, header, parts, worker_queue):
        msg_number = header.get("msg_number", -1)
        try:
            time_start = time.perf_counter()
            filename = header['filename']
            saveraw = True
            if len(parts) > 1:
                saveraw = parts[1].get("save_raw", True)
            if filename and saveraw:
                if os.path.isfile(filename):
                    time_isfile = time.perf_counter() - time_start
                    self._fh = h5py.File(filename, 'a')
                    time_opened = time.perf_counter() - time_start
                    logger.info("h%d: opened existing file %s to append, isfile took %lf, with open took %lf", msg_number, filename, time_isfile, time_opened)
                else:
                    self._fh = h5py.File(filename, 'w')
                    time_opened = time.perf_counter() - time_start
                    end = self._dset_name.rfind('/')
                    group_name = self._dset_name[:end]
                    group = self._fh.create_group(group_name)
                    group.attrs['NX_class'] = 'NXdetector'
                    if len(parts) > 1:
                        self.save_dict_to_h5(parts[1], group)
                    group.create_dataset("sequence_number", (0,), maxshape=(None, ), dtype=np.uint32)
                    self._number_dset_name = f"{group_name}/sequence_number"
                    time_dscreated = time.perf_counter() - time_start
                    logger.info("h%d: created new file %s with dataset %s, open took %lf, with created took %lf", msg_number, filename, group_name, time_opened, time_dscreated)
            else:
                self._fh = None
                logger.info("no file opened (live view only)")
            status = {'htype': 'status',
                      'state': 'running'}
            logger.info('h%d: send status running msg', msg_number)
            worker_queue.put([status,])
        except Exception as e:
            status = {'htype': 'status',
                      'state': 'error',
                      'error': str(e)}
            logger.error('h%d: send status error msg %s', msg_number, str(e))
            worker_queue.put([status,])
            
    def _handle_end(self, header, worker_queue):
        msg_number = header.get("msg_number", -1)
        if self._fh:
            self._fh.close()
        status = {'htype': 'status',
                  'state': 'idle'}
        logger.info('h%d: send status idle msg', msg_number)
        worker_queue.put([status,])
    
    def _handle_frame(self, header, parts):
        if self._fh:
            dset = self._fh.get(self._dset_name)
            if not dset:
                if 'chunks' in header:
                    chunks = (1, *header['chunks'])
                else:
                    chunks = (1, *header['shape'])
                    
                compression, compression_opts = None, None
                if header['compression'] == 'bslz4':
                    compression=bitshuffle.BSHUF_H5FILTER
                    compression_opts=(0, bitshuffle.BSHUF_H5_COMPRESS_LZ4)
                    
                dset = self._fh.create_dataset(self._dset_name, 
                                               dtype=header['type'], 
                                               shape=(0, *header['shape']), 
                                               maxshape=(None, *header['shape']),
                                               chunks=chunks,
                                               compression=compression,
                                               compression_opts=compression_opts)
                logger.info("created dataset %s", self._dset_name)
            if "frame" in header and self._number_dset_name:
                ndset = self._fh.get(self._number_dset_name)
                if not ndset:
                    ndset = self._fh.create_dataset(self._number_dset_name, (0,), maxshape=(None,), dtype=np.uint32)
                length = ndset.shape[0]
                ndset.resize(length + 1, axis=0)
                ndset[length] = int(header["frame"])
            n = dset.shape[0]
            dset.resize(n+1, axis=0)
            offsets = [n, *[0]*(dset.ndim-1)]
            for i in range(1, len(parts)):
                offsets[1] = i - 1
                dset.id.write_direct_chunk(offsets, parts[i])
            logger.debug("wrote frame at offsets %s", offsets)
