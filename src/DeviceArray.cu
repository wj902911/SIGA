#include <DeviceArray.h>

__global__
void inplaceAddNestedArrayKernel(DeviceNestedArrayView<double> arrayA, DeviceNestedArrayView<double> arrayB)
{
    int totalSize = arrayA.totalSize();
    for (int arrayIdx = blockIdx.x * blockDim.x + threadIdx.x; 
        arrayIdx < totalSize; arrayIdx += blockDim.x * gridDim.x)
    {
        arrayA.data()[arrayIdx] += arrayB.data()[arrayIdx];
    }
}

void DeviceNestedArrayView<double>::operator+=(DeviceNestedArrayView<double> other)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (this->totalSize() + threadsPerBlock - 1) / threadsPerBlock;
    inplaceAddNestedArrayKernel<<<blocksPerGrid, threadsPerBlock>>>
        (*this, other);
}