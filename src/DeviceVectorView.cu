#include <DeviceVectorView.h>
#include <cassert>

__global__
void normKernel(double *data, int size, double* result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < size; idx += blockDim.x * gridDim.x)
    {
        atomicAdd(result, data[idx] * data[idx]);
    }
}

__host__
double DeviceVectorView<double>::norm() const
{
    double* d_result;
    double h_result = 0.0;
    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(double));
    assert(err == cudaSuccess && "cudaMalloc failed in DeviceVectorView::norm");
    err = cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "cudaMemcpy failed in DeviceVectorView::norm");
    int blockSize = 256;
    int gridSize = (this->size() + blockSize - 1) / blockSize;
    normKernel<<<gridSize, blockSize>>>(this->data(), this->size(), d_result);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cudaDeviceSynchronize failed in DeviceVectorView::norm");
    err = cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in DeviceVectorView::norm");
    err = cudaFree(d_result);
    assert(err == cudaSuccess && "cudaFree failed in DeviceVectorView::norm");
    return sqrt(h_result);
}