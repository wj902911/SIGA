#include <GaussPoints_d.h>

__global__
void retrieveSizes(int numArrays, const DeviceObjectArray<double>* arrays, 
                  int* sizes)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < numArrays; idx += blockDim.x * gridDim.x)
        sizes[idx] = arrays[idx].size();
}

__global__
void retrieveData(int dir, const DeviceObjectArray<double>* array, 
                  double* data)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < array->size(); idx += blockDim.x * gridDim.x)
        data[idx] = (*array)[idx];
}