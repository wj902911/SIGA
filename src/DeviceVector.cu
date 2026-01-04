#include <DeviceVector.h>

__global__ void destructKernel(DeviceVector<double>* ptr, size_t count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) 
        ptr[idx].~DeviceVector<double>();
}