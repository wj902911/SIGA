#include <DeviceMatrixView.h>

__global__
void inplaceAddKernel(double* dataA, const double* dataB, int rows, int cols)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < rows * cols; idx += blockDim.x * gridDim.x)
    {
        dataA[idx] += dataB[idx];
    }
}

template<>
void DeviceMatrixView<double>::operator+=(DeviceMatrixView<double> other)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows() * cols() + threadsPerBlock - 1) / threadsPerBlock;
    inplaceAddKernel<<<blocksPerGrid, threadsPerBlock>>>
        (this->data(), other.data(), this->rows(), this->cols());
}