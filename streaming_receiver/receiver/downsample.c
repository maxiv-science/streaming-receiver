#include <stdint.h>
#include <stdio.h>
#define min(X,Y) (((X) < (Y)) ? (X) : (Y))

void c_downsample(uint16_t* img, int nrows, int ncols, int factor, uint16_t* output)
{
    int oi = 0;
    int oj = 0;
    int ocols = ncols / factor;
    if (ncols % factor) {
        ocols++;
    }
    for (int i=0; i<nrows; i+=factor) {
        oj = 0;
        for (int j=0; j<ncols; j+=factor) {
            int n = 0;
            float sum = 0.0f;
            for (int k=i; k<min(nrows, i+factor); k++) {
                for (int l=j; l<min(ncols, j+factor); l++) {
                    sum += img[k*ncols + l];
                    n++;
                }
            }
            //printf("%d %d %f %d\n", oi, oj, sum, n);
            output[oi*ocols + oj] = sum / n;
            oj++;
        }
        oi++;
    }
}
