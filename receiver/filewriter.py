import os
import json
import h5py
import bitshuffle
from threading import Thread
from .queuey import Queuey

class FileWriter():
    def __init__(self, dset_name):
        self._fh = None
        self._thread = None
        self._dset_name = dset_name
        
    def run(self, worker_queue: Queuey, writer_queue: Queuey):
        self._thread = Thread(target=self._main, args=(worker_queue, writer_queue))
        self._thread.start()
        
    def _main(self, worker_queue, writer_queue):
        while True:
            parts = writer_queue.get()
            header = parts[0]
            print('writer', header)
            if header['htype'] == 'header':
                self._handle_start(header, parts, worker_queue)
            elif header['htype'] == 'series_end':
                self._handle_end(worker_queue)
            else:
                self._handle_frame(header, parts)
                    
    def _handle_start(self, header, parts, worker_queue):
        try:
            filename = header['filename']
            if filename:
                if os.path.isfile(filename):
                    self._fh = h5py.File(filename, 'a')
                else:
                    self._fh = h5py.File(filename, 'w')
                    end = self._dset_name.rfind('/')
                    group_name = self._dset_name[:end]
                    group = self._fh.create_group(group_name)
                    group.attrs['NX_class'] = 'NXdetector'
                    if len(parts) > 1:
                        for key, value in parts[1].items():
                            group.create_dataset(key, data=value)
            else:
                self._fh = None
            status = {'htype': 'status',
                      'state': 'running'}
            worker_queue.put([status,])
        except Exception as e:
            status = {'htype': 'status',
                      'state': 'error',
                      'error': str(e)}
            worker_queue.put([status,])
            
    def _handle_end(self, worker_queue):
        if self._fh:
            self._fh.close()
        status = {'htype': 'status',
                  'state': 'idle'}
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
                            
            n = dset.shape[0]
            dset.resize(n+1, axis=0)
            offsets = [n, *[0]*(dset.ndim-1)]
            for i in range(1, len(parts)):
                offsets[1] = i - 1
                dset.id.write_direct_chunk(offsets, parts[i])
