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

__global__
void compareKernel(DeviceVectorView<double> arrayA, DeviceVectorView<double> arrayB, DeviceVectorView<int> mismatchIndices, DeviceMatrixView<double> mismatchValues) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < arrayA.size(); idx += blockDim.x * gridDim.x) 
    {
        if (abs(arrayA[idx] - arrayB[idx]) > 1e-6)
        {
            int mismatchIdx = atomicAdd(&mismatchIndices[0], 1);
            mismatchIndices[mismatchIdx + 1] = idx;
            mismatchValues(0, mismatchIdx) = arrayA[idx];
            mismatchValues(1, mismatchIdx) = arrayB[idx];
        }
    }
}

template<>
void DeviceNestedArrayView<double>::operator+=(DeviceNestedArrayView<double> other)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (this->totalSize() + threadsPerBlock - 1) / threadsPerBlock;
    inplaceAddNestedArrayKernel<<<blocksPerGrid, threadsPerBlock>>> (*this, other);
}

template<>
void DeviceArray<double>::compare(const DeviceArray<double>& other) const {
    DeviceArray<int> mismatchIndices(m_size + 1);
    DeviceArray<double> mismatchValues(m_size * 2);

    int minGrid, blockSize;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, compareKernel, 0, m_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during occupancy calculation: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    int gridSize = (m_size + blockSize - 1) / blockSize;
    gridSize = std::min(gridSize, minGrid);
    compareKernel<<<gridSize, blockSize>>>(this->vectorView(),
                                           other.vectorView(),
                                           mismatchIndices.vectorView(),
                                           mismatchValues.matrixView(2, m_size));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    int mismatchCount = mismatchIndices[0];
    if (mismatchCount > 0)
    {
        std::cout << "Number of mismatches: " << mismatchCount << " out of " << m_size << std::endl;
        int outputNum = std::min(mismatchCount, 20);
        std::cout << "The first " << outputNum << " mismatch indices:\n";
        DeviceVectorView<int> mismatchIndicesView(mismatchIndices.data() + 1, outputNum);
        mismatchIndicesView.print();
        std::cout << "The first " << outputNum << " mismatch values:\n";
        mismatchValues.matrixPartialView(2, outputNum).print();
    }
    else
    {
        std::cout << "Arrays are identical." << std::endl;
        return;
    }
    
}