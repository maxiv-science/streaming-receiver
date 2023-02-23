#include <stdint.h>
#include <immintrin.h>

// unpacks mono12p 12bit packed uint to 16bit uint
// uses 24 bytes starting at data by doing a 32-byte load from data-4.
// Make sure the address data-4 is valid before using this function
// uses 24 bytes and generates 32 bytes output so unpacks 16 integer values at a time
void unpack_mono12p_avx(const uint8_t* data, uint16_t* output)
{
    __m256i v = _mm256_loadu_si256( (const __m256i*)(data-4) );
    const __m256i bytegrouping =
        _mm256_setr_epi8(4,5, 5,6,  7,8, 8,9,  10,11, 11,12,  13,14, 14,15, // low half uses last 12B
                         0,1, 1,2,  3,4, 4,5,   6, 7,  7, 8,   9,10, 10,11); // high half uses first 12B
    // rearange bytes in the correct order
    v = _mm256_shuffle_epi8(v, bytegrouping);

    // lo = v & 0x0FFF;
    __m256i lo  = _mm256_and_si256(v, _mm256_set1_epi16(0x0FFF));
    // hi = v >> 4;
    __m256i hi = _mm256_srli_epi16(v, 4);                             
    _mm256_storeu_si256((__m256i*)output, _mm256_blend_epi16(lo, hi, 0b10101010));
}

void unpack_mono12p(const uint8_t* data, int size, uint16_t* output)
{
    int i = 0;
    int j = 0;
    
    // unpack the first 6 bytes normally
    while (j < 6) {
        output[i] = data[j] + ((data[j+1] & 0x0F) << 8);
        output[i+1] = (data[j+1] >> 4) + (data[j+2] << 4);
        i += 2;
        j += 3;
    }
    // use the vectorized version to unpack 16 values at a time and advance by 24 bytes at a time
    while (j < (size-27)) {
        unpack_mono12p_avx(&data[j], &output[i]);
        i += 16;
        j += 24;
    }
    
    // unpack the rest 
    while (j < size) {
        output[i] = data[j] + ((data[j+1] & 0x0F) << 8);
        output[i+1] = (data[j+1] >> 4) + (data[j+2] << 4);
        i += 2;
        j += 3;
    }
} 
