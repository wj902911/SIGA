#include "GPUFunction.h"

__global__
void eval_into_Kernel_displacement(
    int numPoints,
    MultiPatchDeviceView displacement,
    DeviceMatrixView<int> numPointsPerDir,
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
            int patchPoints = 1;
            for (int d = 0; d < displacement.domainDim(); d++)
                patchPoints *= numPointsPerDir(d, i);
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
#if 0
__global__
void GPUFunctionPrintKernel(const MultiPatchDeviceView& displacement)
{
    displacement.print();
}
#endif

GPUDisplacementFunction::GPUDisplacementFunction(const MultiPatchDeviceView &view)
    : m_displacementDeviceView(view)
{
    //GPUFunctionPrintKernel<<<1, 1>>>(m_displacementDeviceView);
    //cudaDeviceSynchronize();
}

void GPUDisplacementFunction::eval_into(DeviceMatrixView<double> gridPoints,
                                        DeviceMatrixView<int> numPointsPerDir,
                                        DeviceMatrixView<double> values) const
{
    int numPoints = gridPoints.cols();
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
        eval_into_Kernel_displacement, 0, numPoints);
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    eval_into_Kernel_displacement<<<gridSize, blockSize>>>(numPoints,
        m_displacementDeviceView, numPointsPerDir, gridPoints, values);
    cudaError_t err = cudaGetLastError();
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

}