#cython: language_level=3

cimport cython
from libc.stdint cimport uint8_t, uint16_t, int32_t

cdef extern void read_cbf(char* cbf, int32_t* output) nogil
cdef extern void unpack_mono12p(const uint8_t* data, int size, uint16_t* output) nogil

def decompress_cbf(char[::1] blob, int32_t[:, ::1] output):
    with nogil:
        read_cbf(&blob[0], &output[0, 0])


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
                if value > 0 and value < max_tot:
                    output[i, j] = tot_tensor[i, j, value]
                else:
                    output[i, j] = -1.0
