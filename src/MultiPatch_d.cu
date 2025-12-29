#include "MultiPatch_d.h"

//template __global__ void testKernel<int>(int);
__global__
void edgeLengthesKernel(const MultiPatch_d* patches, double* patchLengthes)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->totalNumBdGPs(); idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int point_idx = patches->threadPatch_edge(idx, patch);
        int dir(0);
        point_idx = patches->threadEdgeDir(point_idx, patch, dir);
        int edgeIdx(0);
        point_idx = patches->threadEdge(point_idx, patch, dir, edgeIdx);
        int basisDim = patches->getBasisDim();
        Patch_d edge;
        switch (basisDim)
        {
            case 2:
            {
                edge = patches->boundary(patch, edgeIdx + 1);
                break;
            }
            case 3:
            {
                int faceIdx = edgeIdx / 2;
                int localEdgeIdx = edgeIdx % 2;
                Patch_d face = patches->boundary(patch, faceIdx + 1);
                switch (faceIdx)
                {
                    case 0: case 1: case 2: case 3:
                    {
                        edge = face.boundary(localEdgeIdx + 1);
                        break;
                    }
                    case 4: case 5:
                    {
                        edge = face.boundary(localEdgeIdx + 3);
                        break;
                    }
                }
                break;
            }
        }
        DeviceObjectArray<int> numGPs(1);
        numGPs[0]=edge.basis().numGPsInDir(0);
        GaussPoints_d GPs(1, numGPs);
        DeviceVector<double> pt;
        double wt = edge.basis().gsPoint(point_idx, GPs, pt);
        DeviceMatrix<double> activeCPs = edge.getActiveControlPoints(pt);
        DeviceObjectArray<DeviceVector<double>> values;
        edge.basis().evalAllDers_into(pt, 1, values);
        DeviceObjectArray<DeviceMatrix<double>> md;
        md.resize(2);
        md[0] = values[0].transpose() * activeCPs;
        md[1] = values[1].reshape(1, activeCPs.rows()) * activeCPs;
        DeviceMatrix<double> jacobian = md[1].transpose();
        double length = wt * jacobian.norm();
        int edgeIdxOffset = patch*patches->getNumEdgesInPatch(0) + edgeIdx;
        atomicAdd(&patchLengthes[edgeIdxOffset], length);
    }
}

__global__
void totalNumBdGPsKernel(const MultiPatch_d* patches, int* result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->getNumPatches(); idx += blockDim.x * gridDim.x)
    {
        int patchBdGPs = patches->totalNumBdGPsInPatch(idx);
        atomicAdd(result, patchBdGPs);
    }
}

__global__
void getNumPatchesKernel(const MultiPatch_d* patches, int* result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *result = patches->getNumPatches();
    }
}

__global__
void getTotalNumEdgesKernel(const MultiPatch_d* patches, int* result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->getNumPatches(); idx += blockDim.x * gridDim.x)
    {
        int patchEdges = patches->getNumEdgesInPatch(idx);
        atomicAdd(result, patchEdges);
    }
}

__global__
void getBasisDimKernel(const MultiPatch_d* patches, int* result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *result = patches->getBasisDim();
    }
}

__global__
void getPatchLengthesKernel(const MultiPatch_d* patches, const double* edgeLengthes, double* patchLengthes)
{
    // compute patch lengthes in each direction, each thread deal with one edge, its length contribution
    // is its length devided by the number of edges in each direction.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->getTotalNumEdges(); idx += blockDim.x * gridDim.x)
    {
        int threadPatch = 0, point_idx = idx;
        for (int i = 0; i < patches->getNumPatches(); i++)
        {
            int patchEdges = patches->getNumEdgesInPatch(i);
            if (point_idx < patchEdges)
            {
                threadPatch = i;
                //printf("Thread %d processing Patch %d\n", idx, threadPatch);
                break;
            }
            point_idx -= patchEdges;
        }
        int threadDir = 0;
        int dim = patches->getBasisDim();
        for (int d = 0; d < dim; d++)
        {
            int dirEdges = patches->getNumEdgesInEachDir(threadPatch);
            if (point_idx < dirEdges)
            {
                threadDir = d;
                break;
            }
            point_idx -= dirEdges;
        }
        double edgeLengthContribution = edgeLengthes[idx] / patches->getNumEdgesInEachDir(threadPatch);
        //printf("Patch %d, Dir %d, Edge idx %d, Edge length %f, Contribution %f\n", 
        //       threadPatch, threadDir, idx, edgeLengthes[idx], edgeLengthContribution);
        atomicAdd(&patchLengthes[threadPatch * dim + threadDir], edgeLengthContribution);
    }
}

__global__
void getUpperSupportsKernel(const MultiPatch_d* patches, double* upperSupports)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->getNumPatches(); idx += blockDim.x * gridDim.x)
    {
        int basisDim = patches->getBasisDim();
        for (int d = 0; d < basisDim; d++)
        {
            upperSupports[idx * basisDim + d] = patches->patch(idx).upperSupportsInDir(d);
        }
    }
}

__global__
void getLowerSupportsKernel(const MultiPatch_d* patches, double* lowerSupports)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < patches->getNumPatches(); idx += blockDim.x * gridDim.x)
    {
        int basisDim = patches->getBasisDim();
        for (int d = 0; d < basisDim; d++)
        {
            lowerSupports[idx * basisDim + d] = patches->patch(idx).lowerSupportsInDir(d);
        }
    }
}

__global__
void getAllControlPointsKernel(const MultiPatch_d* patches, int dataSize, double* allControlPoints)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < dataSize; idx += blockDim.x * gridDim.x)
    {
        int point_idx = idx;
        int threadPatch = 0;
        for (int i = 0; i < patches->getNumPatches(); i++)
        {
            int patchDataSize = patches->getNumControlPoints(i) * patches->getCPDim();
            if (point_idx < patchDataSize)
            {
                threadPatch = i;
                break;
            }
            point_idx -= patchDataSize;
        }
        allControlPoints[idx] = patches->patch(threadPatch).getControlPoints().data()[point_idx];
    }
}

#if 0
MultiPatch_d::MultiPatch_d(int basisDim, int CPDim, int numPatches, double *knots, 
                           int *numKnots, int *orders, double *controlPoints, 
                           int *numControlPoints, int *numGpAndEle)
    : m_basisDim(basisDim), m_CPDim(CPDim), m_numPatches(numPatches),
      m_knots(knots), m_numKnots(numKnots), m_orders(orders),
      m_controlPoints(controlPoints), m_numControlPoints(numControlPoints),
      m_numGpAndEle(numGpAndEle) { }
#endif


MultiPatch_d::MultiPatch_d(const MultiPatch &mp)
: m_patches(mp.getNumPatches())//, m_topology(mp.topology())
{
#if 1
    for (int i = 0; i < mp.getNumPatches(); i++)
        m_patches.at(i) = Patch_d(mp.patch(i));
#else
    Patch_d* h_patches_d = new Patch_d[mp.getNumPatches()];
    for (int i = 0; i < mp.getNumPatches(); i++)
    {
        Patch_d temp(mp.patch(i));
        //temp.getControlPoints().print();
        h_patches_d[i] = temp;
        //h_patches_d[i].getControlPoints().print();
    }
    m_patches.parallelDataSetting(h_patches_d, mp.getNumPatches());
    delete[] h_patches_d;
#endif
}

#if 0
int MultiPatch_d::threadPatch(int idx, int &patch) const
{
    int point_idx = idx;
    for (int i = 0; i < m_patches.size(); i++)
    {
        int patch_points = m_patches[i].basis().totalNumGPs();
        if (point_idx < patch_points) 
        {
            patch = i;
            break;
        }
        point_idx -= patch_points;
    }
    return point_idx;
}

double MultiPatch_d::gsPoint(int idx, int patch, const GaussPoints_d &gps, 
                             DeviceVector<double> &result) const
{
    return m_patches[patch].basis().gsPoint(idx, gps, result);
}

double MultiPatch_d::gsPoint(int idx, const DeviceObjectArray<GaussPoints_d> &gps, 
                             DeviceVector<double> &result) const
{
    int patch = 0;
    int point_idx = threadPatch(idx, patch);
    return gsPoint(point_idx, patch, gps[patch], result);
}

void MultiPatch_d::evalAllDers_into(int patch, int dir, double u, int n, 
                                    DeviceObjectArray<DeviceVector<double>> &result) const
{
    m_patches[patch].basis().evalAllDers_into(u, dir, n, result);
}

void MultiPatch_d::evalAllDers_into(int patch, const DeviceVector<double> &u, int n, 
                                    DeviceObjectArray<DeviceVector<double>> &result) const
{
    m_patches[patch].basis().evalAllDers_into(u, n, result);
}
#endif

MultiPatch_d::MultiPatch_d(int numPatches)
: m_patches(numPatches) {}

void MultiPatch_d::getEdgeLengthes(DeviceVector<double> &lengths) const
{
    lengths.resize(getTotalNumEdges_host());
    lengths.setZero();

    int totalBDGPs = totalNumBdGPs_host();
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       edgeLengthesKernel, 0, totalBDGPs);
    int gridSize = (totalBDGPs + blockSize - 1) / blockSize;
    edgeLengthesKernel<<<gridSize, blockSize>>>(this, lengths.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during edgeLengthesKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (edgeLengthesKernel): %s\n", 
               cudaGetErrorString(err));
    }
}

int MultiPatch_d::totalNumBdGPs_host() const
{
    int h_result = 0;
    int* d_result;
    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in totalNumBdGPs_host");
    err = cudaMemset(d_result, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in totalNumBdGPs_host");
    int minGrid, blockSize, numPatches = getNumPatches_host();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       totalNumBdGPsKernel, 0, numPatches);
    int gridSize = (numPatches + blockSize - 1) / blockSize;
    totalNumBdGPsKernel<<<gridSize, blockSize>>>(this, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during totalNumBdGPsKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (totalNumBdGPsKernel): %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in totalNumBdGPs_host");
    cudaFree(d_result);
    return h_result;
}

int MultiPatch_d::getNumPatches_host() const
{
    int h_result = 0;
    int* d_result;
    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in getNumPatches_host");
    err = cudaMemset(d_result, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in getNumPatches_host");
    getNumPatchesKernel<<<1, 1>>>(this, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getNumPatchesKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getNumPatchesKernel): %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in getNumPatches_host");
    cudaFree(d_result);
    return h_result;
}

int MultiPatch_d::getTotalNumEdges_host() const
{
    int h_result = 0;
    int* d_result;
    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in getTotalNumEdges_host");
    err = cudaMemset(d_result, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in getTotalNumEdges_host");
    int minGrid, blockSize, numPatches = getNumPatches_host();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       getTotalNumEdgesKernel, 0, numPatches);
    int gridSize = (numPatches + blockSize - 1) / blockSize;
    getTotalNumEdgesKernel<<<gridSize, blockSize>>>(this, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getTotalNumEdgesKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getTotalNumEdgesKernel): %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in getTotalNumEdges_host");
    cudaFree(d_result);
    return h_result;
}

void MultiPatch_d::getPatchLengthes(DeviceVector<double> &lengths) const
{
    DeviceVector<double> edgeLengths;
    getEdgeLengthes(edgeLengths);
    //edgeLengths.print();
    //printf("\n");
    lengths.resize(getNumPatches_host() * getBasisDim_host());
    lengths.setZero();
    int minGrid, blockSize, totalEdges = getTotalNumEdges_host();
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       getPatchLengthesKernel, 0, totalEdges);
    int gridSize = (totalEdges + blockSize - 1) / blockSize;
    getPatchLengthesKernel<<<1, 1>>>(this, edgeLengths.data(), lengths.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getPatchLengthesKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getPatchLengthesKernel): %s\n", 
               cudaGetErrorString(err));
    }
}

int MultiPatch_d::getBasisDim_host() const
{
    int h_result = 0;
    int* d_result;
    cudaError_t err = cudaMalloc((void**)&d_result, sizeof(int));
    assert(err == cudaSuccess && "cudaMalloc failed in getBasisDim_host");
    err = cudaMemset(d_result, 0, sizeof(int));
    assert(err == cudaSuccess && "cudaMemset failed in getBasisDim_host");
    getBasisDimKernel<<<1, 1>>>(this, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getBasisDimKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getBasisDimKernel): %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed in getBasisDim_host");
    cudaFree(d_result);
    return h_result;
}

void MultiPatch_d::getUpperSupports(DeviceMatrix<double> &upperSupports) const
{
    int numPatches = getNumPatches_host();
    upperSupports.resize(numPatches, getBasisDim_host());
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       getUpperSupportsKernel, 0, numPatches);
    int gridSize = (numPatches + blockSize - 1) / blockSize;
    getUpperSupportsKernel<<<gridSize, blockSize>>>(this, upperSupports.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getUpperSupportsKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getUpperSupportsKernel): %s\n", 
               cudaGetErrorString(err));
    }
}

void MultiPatch_d::getLowerSupports(DeviceMatrix<double> &lowerSupports) const
{
    int numPatches = getNumPatches_host();
    lowerSupports.resize(numPatches, getBasisDim_host());
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       getLowerSupportsKernel, 0, numPatches);
    int gridSize = (numPatches + blockSize - 1) / blockSize;
    getLowerSupportsKernel<<<gridSize, blockSize>>>(this, lowerSupports.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getLowerSupportsKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getLowerSupportsKernel): %s\n", 
               cudaGetErrorString(err));
    }
}

void MultiPatch_d::retrieveControlPoints(MultiPatch &mp) const
{
    int dataSize = 0;
    for (int i = 0; i < mp.getNumPatches(); i++)
    {
        dataSize += mp.patch(i).getControlPoints().size();
    }
    DeviceVector<double> allCPs(dataSize);
    allCPs.setZero();
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
                                       getAllControlPointsKernel, 0, dataSize);
    int gridSize = (dataSize + blockSize - 1) / blockSize;
    getAllControlPointsKernel<<<gridSize, blockSize>>>(this, dataSize, allCPs.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during getAllControlPointsKernel launch: %s\n", 
               cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error during device synchronization (getAllControlPointsKernel): %s\n", 
               cudaGetErrorString(err));
    }
    Eigen::VectorXd h_allCPs(dataSize);
    err = cudaMemcpy(h_allCPs.data(), allCPs.data(), dataSize * sizeof(double), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error during cudaMemcpy in retrieveControlPoints: %s\n", 
               cudaGetErrorString(err));
    }
    int offset = 0;
    for (int i = 0; i < mp.getNumPatches(); i++)
    {
        int patchDataSize = mp.patch(i).getControlPoints().size();
        Eigen::MatrixXd cpMat = Eigen::Map<const Eigen::MatrixXd>(h_allCPs.data() + offset, 
                                                                  mp.patch(i).getCPDim(), 
                                                                  patchDataSize / mp.patch(i).getCPDim());
        mp.patch(i).setControlPoints(cpMat.transpose());
        offset += patchDataSize;
    }
}
