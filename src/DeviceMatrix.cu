#include "DeviceMatrix.h"

__global__
void parallPlus(double* a, double* b, double* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        c[idx] = a[idx] + b[idx];
    }
}