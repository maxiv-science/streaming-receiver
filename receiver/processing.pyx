#cython: language_level=3

cimport cython
import numpy as np
from libc.stdint cimport uint8_t, uint16_t, int32_t
from cpython cimport PyObject_GetBuffer, \
                     PyBUF_ANY_CONTIGUOUS, \
                     PyBuffer_Release

cdef extern void read_cbf(char* cbf, int32_t* output) nogil
cdef extern void c_unpack_mono12p(const uint8_t* data, int size, uint16_t* output) nogil
cdef extern void c_downsample(uint16_t* img, int nrows, int ncols, int factor, uint16_t* output) nogil

def decompress_cbf(char[::1] blob, int32_t[:, ::1] output):
    with nogil:
        read_cbf(&blob[0], &output[0, 0])
        
def unpack_mono12p(uint8_t[::1] data, int size, uint16_t[:, ::1] output):
    c_unpack_mono12p(&data[0], size, &output[0, 0])
    
@cython.boundscheck(False)
@cython.wraparound(False)
def convert_tot(int32_t[:, ::1] tot, double[:, :, ::1] tot_tensor, float[:, ::1] output):
    cdef Py_ssize_t rows = tot.shape[0]
    cdef Py_ssize_t cols = tot.shape[1]
    cdef int32_t max_tot = tot_tensor.shape[2]
    cdef int32_t value
    
    with nogil:
        for i in range(rows):
            for j in range(cols):
                value = tot[i, j]
                if value >= 0 and value < max_tot:
                    output[i, j] = tot_tensor[i, j, value]
                elif value >= max_tot:
                    output[i, j] = -2.0
                else:
                    output[i, j] = -1.0
   
def downsample(img, shape, int factor):
    cdef int rows = shape[0]
    cdef int cols = shape[1]
    
    m = rows // factor
    if rows % factor:
        m += 1
        
    n = cols // factor
    if cols % factor:
        n += 1
    
    cdef uint16_t[:,:] output = np.empty((m, n), dtype=np.uint16)
    cdef Py_buffer view
    PyObject_GetBuffer(img, &view, PyBUF_ANY_CONTIGUOUS)
    c_downsample(<uint16_t*>view.buf, rows, cols, factor, &output[0, 0])
    PyBuffer_Release(&view)
    return output.base
