#include "GPUFunction.h"

__global__
void eval_into_Kernel_displacement(
    int numPoints,
    MultiPatchDeviceView displacement,
    DeviceVectorView<int> numPointsPerPatch,
    DeviceMatrixView<double> gridPoints,
    DeviceMatrixView<double> values)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < numPoints; idx += blockDim.x * gridDim.x)
    {
        //printf("Kernel thread %d started\n", idx);
        //displacement.print();
        int point_idx = idx;
        int threadPatch = 0;
        for (int i = 0; i < displacement.numPatches(); i++)
        {
            int patchPoints = numPointsPerPatch[i];
            if (point_idx < patchPoints)
            {
                threadPatch = i;
                break;
            }
            point_idx -= patchPoints;
        }
        //printf("Thread %d processing point %d in patch %d\n", 
        //       idx, point_idx, threadPatch);
        double gridPointData[3] = {0}; // Assuming max dimension is 3
        DeviceVectorView<double> gridPoint(gridPointData, displacement.domainDim());
        for (int d = 0; d < displacement.domainDim(); d++)
            gridPoint[d] = gridPoints(d, idx);
        double valueData[3] = {0}; // Assuming max dimension is 3
        DeviceVectorView<double> value(valueData, displacement.targetDim());

        //printf("Thread %d evaluating at point:\n", idx);
        //gridPoint.print();

        displacement.patch(threadPatch).evaluate(gridPoint, value);
        //printf("Thread %d got value:\n", idx);
        //value.print();
        for (int d = 0; d < displacement.targetDim(); d++)
            values(d, idx) = value[d];
    }
}

__global__
void eval_into_Kernel_displacement_blockPerPoint(
    int numPoints, int numBlocksPerPoint, int numActivePerBlock,
    MultiPatchDeviceView displacement,
    DeviceVectorView<int> numPointsPerPatch,
    DeviceMatrixView<double> gridPoints,
    DeviceMatrixView<double> values)
{
    extern __shared__ double shmem[];
    int totalNumBlocks = numPoints * numBlocksPerPoint;
    for (int bidx = blockIdx.x; bidx < totalNumBlocks; bidx += gridDim.x)
    {
        //printf("Kernel thread %d started\n", bidx);
        //displacement.print();
        __shared__ int threadPatch, /*numThreadsPerBlock,*/ blockCoord, idx;
        __shared__ double gridPointData[3], valueData[3]; // Assuming max dimension is 3
        DeviceVectorView<double> gridPoint(gridPointData, displacement.domainDim());
        DeviceVectorView<double> value(valueData, displacement.targetDim());
        int threadId = threadIdx.x;
        if (threadId == 0)
        {
            //numThreadsPerBlock = blockDim.x;
            idx = bidx;
            blockCoord = idx % numBlocksPerPoint;
            idx /= numBlocksPerPoint;
            int point_idx = idx;
            for (int i = 0; i < displacement.numPatches(); i++)
            {
                int patchPoints = numPointsPerPatch[i];
                if (point_idx < patchPoints)
                {
                    threadPatch = i;
                    break;
                }
                point_idx -= patchPoints;
            }
            
            //printf("bidx=%d, idx=%d, blockCoord=%d, threadPatch=%d\n", bidx, idx, blockCoord, threadPatch);
        }
        for (int d = threadId; d < displacement.domainDim(); d += blockDim.x)
            gridPoint[d] = gridPoints(d, idx);
        for (int d = threadId; d < displacement.domainDim(); d += blockDim.x)
            value[d] = 0.0;
        __syncthreads();
        TensorBsplineBasisDeviceView basis = displacement.basis(threadPatch);
        int P = basis.knotsOrder(0);
        int dim = basis.dim();
#if 0
        if (threadId == 0)
        {
            gridPoint.print();
            printf("\n");
        }
#endif
        //printf("Thread %d processing point %d in patch %d\n", 
        //       idx, point_idx, threadPatch);
        

        //printf("Thread %d evaluating at point:\n", threadId);
        //gridPoint.print();
        DeviceMatrixView<double> basisValuesAndDers(shmem, P+1, dim);
        //printf("%d\n", basisValuesAndDers.size());
        if (threadId < dim)
            //basis.evalAllDers_into(threadId, blockDim.x, gridPoint, 0, basisValuesAndDers);
            basis.evalAllDers_into(threadId, blockDim.x, gridPoint, 0, shmem + (P+1) * dim, basisValuesAndDers);
        __syncthreads();

#if 0
        if (threadId == 0)
        {
            basisValuesAndDers.print();
            printf("\n");
        }
#endif

        displacement.patch(threadPatch).evaluate(threadId, blockDim.x, blockCoord, 
            numActivePerBlock, basisValuesAndDers, gridPoint, value);
        __syncthreads();
        //printf("Thread %d got value:\n", threadId);
        //value.print();
        for (int d = threadId; d < displacement.targetDim(); d += blockDim.x)
            atomicAdd(&values(d, idx), value[d]);
    }
}
#if 0
__global__
void GPUFunctionPrintKernel(const MultiPatchDeviceView& displacement)
{
    displacement.print();
}
#endif

GPUDisplacementFunction::GPUDisplacementFunction(const MultiPatch &displacementHost)
    : m_displacementHost(displacementHost), m_displacementDeviceData(displacementHost)
{
    //GPUFunctionPrintKernel<<<1, 1>>>(m_displacementDeviceView);
    //cudaDeviceSynchronize();
}

void GPUDisplacementFunction::eval_into(DeviceMatrixView<double> gridPoints,
                                        DeviceVectorView<int> numPointsPerPatch,
                                        DeviceMatrixView<double> values) const
{
    int numPoints = gridPoints.cols();
    cudaError_t err;
    int N_D = m_displacementHost.numActive();
    int blockSize = std::min(256, N_D);
    int numBlocksPerPoint = (N_D + blockSize - 1) / blockSize;
    int gridSize = numPoints * numBlocksPerPoint;
    int p1 = m_displacementHost.knotOrder() + 1;
    int dim = m_displacementHost.getBasisDim();
    //int numDouble = p1 * dim;
    int numDouble = p1 * dim + (p1 * p1 + 4 * p1) * dim;
    size_t shmemBytes = numDouble * sizeof(double);
    eval_into_Kernel_displacement_blockPerPoint<<<gridSize, blockSize, shmemBytes>>>(numPoints, numBlocksPerPoint, 
        blockSize, m_displacementDeviceData.deviceView(), numPointsPerPatch, gridPoints, values);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in eval_into_Kernel_displacement_blockPerPoint: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Synchronization Error in eval_into_Kernel_displacement_blockPerPoint: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
#if 0
    values.print();
    std::cout << std::endl;

    err = cudaMemset(values.data(), 0, values.size() * sizeof(double));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in cudaMemset for values: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    values.print();
    std::cout << std::endl;

    int minGrid;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        eval_into_Kernel_displacement, 0, numPoints);
    gridSize = (numPoints + blockSize - 1) / blockSize;
    eval_into_Kernel_displacement<<<gridSize, blockSize>>>(numPoints,
        m_displacementDeviceData.deviceView(), numPointsPerPatch, gridPoints, values);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in eval_into_Kernel_displacement: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Synchronization Error in eval_into_Kernel_displacement: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    values.print();
    std::cout << std::endl;
#endif
}