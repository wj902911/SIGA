#include <Patch_d.h>

__global__ void destructKernel(Patch_d* ptr, size_t count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) 
        ptr[idx].~Patch_d();
}