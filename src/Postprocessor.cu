#include "Postprocessor.h"
#include <DeviceObjectPointer.h>

__global__
void distributePointsKernel(const MultiPatch_d* patches, const double* patchLengthes, const int* numPoints, int* numPointsPerDir)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->getNumPatches(); idx += blockDim.x * gridDim.x)
    {
        double volume = 1.0;
        int dim = patches->getBasisDim();
        for (int d = 0; d < dim; d++)
            volume *= patchLengthes[idx * dim + d];
        double unit = pow(numPoints[idx] / volume, 1.0 / dim);
        for (int d = 0; d < dim; d++)
        {
            int dir = d;
            if (d == 0)
                dir = 1;
            else if (d == 1)
                dir = 0;
            numPointsPerDir[idx * dim + d] = ceil(unit*patchLengthes[idx * dim + dir] > 1 ? ceil(unit*patchLengthes[idx * dim + dir]) : 2);
        }
    }
    

}

void PostProcessor::distributePoints(const Eigen::VectorXi &numPoints, Eigen::MatrixXi& numPointsPerDir) const
{
    int dim = m_geometry.getBasisDim();
    int numPatches = m_geometry.getNumPatches();
    numPointsPerDir.resize(dim, numPatches);
    numPointsPerDir.setZero();
    DeviceMatrix<int> numPointsPerDir_d(numPatches, dim);
    numPointsPerDir_d.setZero();

    DeviceVector<double> patchLengthes;
    MultiPatch_d geometry_d(m_geometry);
    MultiPatch_d* d_geometry;
    cudaMalloc((void**)&d_geometry, sizeof(MultiPatch_d));
    cudaMemcpy(d_geometry, &geometry_d, sizeof(MultiPatch_d), cudaMemcpyHostToDevice);
    d_geometry->getPatchLengthes(patchLengthes);
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, distributePointsKernel, 0, numPatches);
    int gridSize = (numPatches + blockSize - 1) / blockSize;
    DeviceVector<int> d_numPoints = numPoints;
    distributePointsKernel<<<gridSize, blockSize>>>(d_geometry, patchLengthes.data(), d_numPoints.data(), numPointsPerDir_d.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in distributePointsKernel: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Synchronization Error in distributePointsKernel: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(numPointsPerDir.data(), numPointsPerDir_d.data(), sizeof(int) * numPatches * dim, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Memcpy Error in distributePointsKernel: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaFree(d_geometry);
}

void PostProcessor::evalGeometryAtPoints(const Eigen::MatrixXi &numPointsPerDir, Eigen::MatrixXd &values) const
{
    int dim = m_geometry.getCPDim();
    int numPoints = 0;
    int numPatches = m_geometry.getNumPatches();
    for (int p = 0; p < numPatches; p++)
        numPoints += numPointsPerDir.col(p).prod();
    //std::cout << "Total number of points: " << numPoints << std::endl;
    values.resize(dim, numPoints);
    values.setZero();
    MultiPatch_d geometry_d(m_geometry);
    DeviceObjectPointer<MultiPatch_d> d_geometry(geometry_d);
    d_geometry.pointer()->eval_into(numPointsPerDir, values);
}
