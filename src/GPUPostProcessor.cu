#include <GPUPostProcessor.h>
#include <fstream>
#include <iomanip>

__global__
void patchLengthesKernel(int numTotalBoundaryGPs,
                         MultiPatchDeviceView geometry, 
                         MultiGaussPointsDeviceView gspts,
                         DeviceVectorView<double> patchLengthes)
                         //DeviceVectorView<int> numPointsPerPatch,
                         //DeviceVectorView<int> numPointsPerDir)
{
   for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numTotalBoundaryGPs; idx += blockDim.x * gridDim.x)
    {
        int patch(0);
        int point_idx = geometry.threadPatch_edge(idx, patch);
        PatchDeviceView geoPatch = geometry.patch(patch);
        TensorBsplineBasisDeviceView geoBasis = geometry.basis(patch);
        int dir(0);
        point_idx = geometry.threadEdgeDir(point_idx, patch, dir);
        int edgeIdx(0);
        point_idx = geometry.threadEdge(point_idx, patch, dir, edgeIdx);
        //printf("Gauss points:\n");
        //gspts.print();
        //printf("Patch %d, dir %d, Edge %d, Point idx %d\n", patch, dir, edgeIdx, point_idx);
        int dim = geometry.domainDim();
        switch (dim)
        {
        case 2:
            {
                //BoxSide_d s(edgeIdx + 1);
                //int dir = s.direction();
                int d = 1 - dir;
                double pt(0);
                double wt = geoBasis.gsPoint(point_idx, d, gspts[patch], pt);
                //printf("Gauss point: %f, weight: %f\n", pt, wt);
                double geoJacobianData[2] = {0.0};
                DeviceVectorView<double> geoJacobian(geoJacobianData, dim);
                geoPatch.boundaryJacobian(edgeIdx + 1, 
                    OneElementDeviceVectorView<double>(&pt), geoJacobian);
                //printf("Geometric Jacobian:\n");
                //geoJacobian.print();
                double length = wt * geoJacobian.norm_device();
                //printf("Length contribution: %f\n", length);
                int edgeIdxOffset = patch * 2 + d;
                atomicAdd(&patchLengthes[edgeIdxOffset], length / 2.0);
                //__syncthreads();
                //if (idx < patchLengthes.size())
                //    patchLengthes[idx] /= 2.0;
                break;
            }
        case 3:
            {
                int d = dir;
                double pt(0);
                double wt = geoBasis.gsPoint(point_idx, d, gspts[patch], pt);
                double geoJacobianData[3] = {0.0};
                DeviceVectorView<double> geoJacobian(geoJacobianData, dim);
                switch(edgeIdx)
                {
                case 0:
                        geoPatch.boundaryJacobian(3, 5, pt, geoJacobian);
                        break;
                case 1:
                        geoPatch.boundaryJacobian(3, 6, pt, geoJacobian);
                        break;
                case 2:
                        geoPatch.boundaryJacobian(4, 5, pt, geoJacobian);
                        break;
                case 3:                        
                        geoPatch.boundaryJacobian(4, 6, pt, geoJacobian);
                        break;
                case 4:
                        geoPatch.boundaryJacobian(1, 5, pt, geoJacobian);
                        break;
                case 5:
                        geoPatch.boundaryJacobian(1, 6, pt, geoJacobian);
                        break;
                case 6:
                        geoPatch.boundaryJacobian(2, 5, pt, geoJacobian);
                        break;
                case 7:
                        geoPatch.boundaryJacobian(2, 6, pt, geoJacobian);
                        break;
                case 8:
                        geoPatch.boundaryJacobian(1, 3, pt, geoJacobian);
                        break;
                case 9:
                        geoPatch.boundaryJacobian(1, 4, pt, geoJacobian);
                        break;
                case 10:                        
                        geoPatch.boundaryJacobian(2, 3, pt, geoJacobian);
                        break;
                case 11:
                        geoPatch.boundaryJacobian(2, 4, pt, geoJacobian);
                        break;
                }
                double length = wt * geoJacobian.norm_device();
                int edgeIdxOffset = patch * 3 + d;
                atomicAdd(&patchLengthes[edgeIdxOffset], length / 4.0);
                break;
            }
        default:
            break;
        }
#if 0
        __syncthreads();
        if (idx < numPointsPerPatch.size())
        {
            double volume = 1.0;
            for (int d = 0; d < dim; d++)
                volume *= patchLengthes[idx * dim + d];
            double unit = pow(numPointsPerPatch[idx] / volume, 1.0 / dim);
            for (int d = 0; d < dim; d++)
            {
                int dir = d;
                if (d == 0)
                    dir = 1;
                else if (d == 1)
                    dir = 0;
                numPointsPerDir[idx * dim + d] = 
                    ceil(unit*patchLengthes[idx * dim + dir] > 1 ? 
                    ceil(unit*patchLengthes[idx * dim + dir]) : 2);
            }
        }
#endif
    }
}

__global__
void distributePointsKernel(int domainDim,
                            DeviceVectorView<double> patchLengthes,
                            DeviceVectorView<int> numPointsPerPatch,
                            DeviceMatrixView<int> numPointsPerDir)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < numPointsPerPatch.size(); idx += blockDim.x * gridDim.x)
    {
        double volume = 1.0;
        int dim = domainDim;
        for (int d = 0; d < dim; d++)
            volume *= patchLengthes[idx * dim + d];
        double unit = pow(numPointsPerPatch[idx] / volume, 1.0 / dim);
        for (int d = 0; d < dim; d++)
        {
            numPointsPerDir(d, idx) = 
                ceil(unit*patchLengthes[idx * dim + d] > 1 ? 
                ceil(unit*patchLengthes[idx * dim + d]) : 2);
        }
    }
}

__global__
void pointGridKernel(int numPoints,
                     int domainDim,
                     DeviceMatrixView<int> numPointsPerDir,
                     DeviceMatrixView<double> pointGrid)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < numPoints; idx += blockDim.x * gridDim.x)
    {
        int point_idx = idx;
        int threadPatch = 0;
        for (int i = 0; i < numPointsPerDir.cols(); i++)
        {
            int patchPoints = 1;
            for (int d = 0; d < domainDim; d++)
                patchPoints *= numPointsPerDir(d, i);
            if (point_idx < patchPoints)
            {
                threadPatch = i;
                break;
            }
            point_idx -= patchPoints;
        }
        double paramData[3] = {0.0}; //max 3D
        DeviceVectorView<double> param(paramData, domainDim);
        int residual = point_idx;
        for (int d = 0; d < domainDim; d++)
        {
            int numPointsInDir = numPointsPerDir(d, threadPatch);
            param[d] = 
                double(residual % numPointsInDir) / double(numPointsInDir - 1);
            residual /= numPointsInDir;
        }
        for (int d = 0; d < domainDim; d++)
            pointGrid(d, idx) = param[d];
    }
}

__global__
void evalAtDistributedPointsKernel(MultiPatchDeviceView geometry,
                                   DeviceMatrixView<int> numPointsPerDir,
                                   DeviceMatrixView<double> pointGrid,
                                   DeviceMatrixView<double> results)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < results.cols(); idx += blockDim.x * gridDim.x)
    {
        int point_idx = idx;
        int threadPatch = 0;
        for (int i = 0; i < geometry.numPatches(); i++)
        {
            int patchPoints = 1;
            for (int d = 0; d < geometry.domainDim(); d++)
                patchPoints *= numPointsPerDir(d, i);
            if (point_idx < patchPoints)
            {
                threadPatch = i;
                break;
            }
            point_idx -= patchPoints;
        }
        double paramData[3] = {0.0}; //max 3D
        DeviceVectorView<double> param(paramData, geometry.domainDim());
        int residual = point_idx;
        for (int d = 0; d < geometry.domainDim(); d++)
        {
            int numPointsInDir = numPointsPerDir(d, threadPatch);
            param[d] = 
                double(residual % numPointsInDir) / double(numPointsInDir - 1);
            residual /= numPointsInDir;
        }
        for (int d = 0; d < geometry.domainDim(); d++)
            pointGrid(d, idx) = param[d];
        double tempData[3] = {0.0}; //max target dim 3
        DeviceVectorView<double> temp(tempData, geometry.targetDim());
        geometry.patch(threadPatch).evaluate(param, temp);
        for (int i = 0; i < geometry.targetDim(); i++)
            results(i, idx) = temp[i];
    }
    //results.print();
}

__global__
void computeMeshKernel(DeviceNestedArrayView<double> breaks,
                       MultiPatchDeviceView geometry,
                       DeviceVectorView<int> meshPointPatchOffsets,
                       DeviceVectorView<int> meshEdgesPatchOffsets,
                       DeviceMatrixView<double> meshPointGrid,
                       DeviceMatrixView<double> meshGeoPoints,
                       DeviceMatrixView<int> meshEdges,
                       DeviceVectorView<int> pointCounters,
                       DeviceVectorView<int> edgeCounters,
                       int numMidPoints, int numMeshCorners)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numMeshCorners; idx += blockDim.x * gridDim.x)
    {
        int point_idx = idx;
        int threadPatch = 0;
        for (int i = 0; i < geometry.numPatches(); i++)
        {
            int patchPoints = 1;
            for (int d = 0; d < geometry.domainDim(); d++)
                patchPoints *= (breaks[i * geometry.domainDim() + d].size());
            if (point_idx < patchPoints)
            {
                threadPatch = i;
                break;
            }
            point_idx -= patchPoints;
        }
        int PointPatchOffset = meshPointPatchOffsets[threadPatch];
        int c_index = PointPatchOffset + point_idx;
        //printf("Thread %d, Patch %d, Point idx %d\n", idx, threadPatch, point_idx);
        int numCornersData[3] = {0}; //max 3D
        DeviceVectorView<int> numCorners(numCornersData, geometry.domainDim());
        int meshCornerCoodData[3] = {0}; //max 3D
        DeviceVectorView<int> meshCornerCoord(meshCornerCoodData, geometry.domainDim());
        int residual = point_idx;
        for (int d = 0; d < geometry.domainDim(); d++)
        {
            numCorners[d] = breaks[threadPatch * geometry.domainDim() + d].size();
            meshCornerCoord[d] = residual % numCorners[d];
            residual /= numCorners[d];
        }
        //printf("Mesh corner coord: (%d, %d)\n", meshCornerCoord[0], meshCornerCoord[1]);
        //int c_index = atomicAdd(pointCounter, 1);
        double paramData[3] = {0.0}; //max 3D
        DeviceVectorView<double> param(paramData, geometry.domainDim());
        for (int d = 0; d < geometry.domainDim(); d++)
        {
            param[d] = breaks[threadPatch * geometry.domainDim() + d][meshCornerCoord[d]];
            meshPointGrid(d, c_index) = param[d];
        }
        //printf("Mesh point param: (%f, %f)\n", param[0], param[1]);
        double geoPointData[3] = {0.0}; //max target dim 3
        DeviceVectorView<double> geoPoint(geoPointData, geometry.targetDim());
        geometry.patch(threadPatch).evaluate(param, geoPoint);
        //printf("Mesh point geo coord: (%f, %f)\n", geoPoint[0], geoPoint[1]);
        for (int d = 0; d < geometry.targetDim(); d++)
            meshGeoPoints(d, c_index) = geoPoint[d];
        int strideData[3] = {1}; //max 3D
        DeviceVectorView<int> stride(strideData, geometry.domainDim());
        for (int d = 0; d < geometry.domainDim(); d++)
        {
            stride[d] = 1;
            for (int dd = 0; dd < d; dd++)
                stride[d] *= numCorners[dd];
        }
        //printf("Mesh corner stride: (%d, %d)\n", stride[0], stride[1]);
        int midIdxStart = numCorners.prod() + PointPatchOffset;
        int edgePatchOffset = meshEdgesPatchOffsets[threadPatch];
        for (int d = 0; d < geometry.targetDim(); d++)
        {
            int neighborCoordData[3] = {0}; //max 3D
            DeviceVectorView<int> neighborCoord(neighborCoordData, geometry.domainDim());
            for (int dd = 0; dd < geometry.domainDim(); dd++)
                neighborCoord[dd] = meshCornerCoord[dd];
            neighborCoord[d] += 1;
            if (neighborCoord[d] < numCorners[d])
            {
                int n_index = c_index + stride[d];
                //printf("Neighbor index: %d\n", n_index);
                double neighborParamData[3] = {0.0}; //max 3D
                DeviceVectorView<double> neighborParam(neighborParamData, geometry.domainDim());
                for (int d = 0; d < geometry.domainDim(); d++)
                    neighborParam[d] = breaks[threadPatch * geometry.domainDim() + d][neighborCoord[d]];
                //printf("Neighbor along dimension %d mesh point param: (%f, %f)\n", 
                //        d, neighborParam[0], neighborParam[1]);
                double midStride = (neighborParam[d] - param[d]) / (numMidPoints + 1);
                int s_index = c_index;
                for(int i = 0; i < numMidPoints; i++)
                {
                    int m_index = atomicAdd(&pointCounters[threadPatch], 1) + midIdxStart;
                    double midParamData[3] = {0.0}; //max 3D
                    DeviceVectorView<double> midParam(midParamData, geometry.domainDim());
                    for (int dd = 0; dd < geometry.domainDim(); dd++)
                    {
                        if (dd == d)
                            midParam[dd] = param[dd] + (i + 1) * midStride;
                        else
                            midParam[dd] = param[dd];
                        meshPointGrid(dd, m_index) = midParam[dd];
                    }
                    //printf("Point %d mid param: (%f, %f)\n", m_index, midParam[0], midParam[1]);
                    double midGeoPointData[3] = {0.0}; //max target dim 3
                    DeviceVectorView<double> midGeoPoint(midGeoPointData, geometry.targetDim());
                    geometry.patch(threadPatch).evaluate(midParam, midGeoPoint);
                    //printf("Point %d mid geo coord: (%f, %f)\n", m_index, midGeoPoint[0], midGeoPoint[1]);
                    for (int dd = 0; dd < geometry.targetDim(); dd++)
                        meshGeoPoints(dd, m_index) = midGeoPoint[dd];
                    int e_index = atomicAdd(&edgeCounters[threadPatch], 1) + edgePatchOffset;
                    //printf("Edge %d: (%d, %d)\n", e_index, s_index, m_index);
                    meshEdges(0, e_index) = s_index - PointPatchOffset;
                    meshEdges(1, e_index) = m_index - PointPatchOffset;
                    s_index = m_index;
                }
                int e_index = atomicAdd(&edgeCounters[threadPatch], 1) + edgePatchOffset;
                //printf("Edge %d: (%d, %d)\n", e_index, s_index, n_index);
                meshEdges(0, e_index) = s_index - PointPatchOffset;
                meshEdges(1, e_index) = n_index - PointPatchOffset;
            }
        }
    }
}

GPUPostProcessor::GPUPostProcessor(const GPUAssembler &assembler, 
                                   const std::vector<int>& numPointsPerPatch,
                                   bool outputMesh, int numMidPoints)
    : m_assembler(assembler), outputMesh(outputMesh)
{
    MultiPatchDeviceView geometry = m_assembler.geometryView();
    int numPatches = geometry.numPatches();
    int domainDim = geometry.domainDim();
    MultiPatch geometryHost = m_assembler.geometryHost();
    int totalBDGPs = geometryHost.totalNumBdGPs();
    MultiGaussPointsDeviceView gspts = m_assembler.gaussPointsView();
    int numTotalLengths = numPatches * domainDim; 
    DeviceArray<double> patchLengthesDeviceData(numTotalLengths);
    DeviceArray<int> numPointsPerPatchDeviceData(numPointsPerPatch);
    m_numPointsPerDir.resize(numPatches * domainDim);
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        patchLengthesKernel, 0, totalBDGPs);
    int gridSize = (numPatches + blockSize - 1) / blockSize;
    patchLengthesKernel<<<gridSize, blockSize>>>(
        totalBDGPs, geometry, gspts,
        patchLengthesDeviceData.vectorView());
        //numPointsPerPatchDeviceData.vectorView(),
        //m_numPointsPerDir.vectorView());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in patchLengthesKernel: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Synchronization Error in patchLengthesKernel: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    //std::vector<double> patchLengthesDirHost;
    //patchLengthesDeviceData.copyToHost(patchLengthesDirHost);
    
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        distributePointsKernel, 0, numPatches);
    gridSize = (numPatches + blockSize - 1) / blockSize;
    distributePointsKernel<<<gridSize, blockSize>>>(
        domainDim,
        patchLengthesDeviceData.vectorView(),
        numPointsPerPatchDeviceData.vectorView(),
        m_numPointsPerDir.matrixView(domainDim, numPatches));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in distributePointsKernel: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Synchronization Error in distributePointsKernel: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    m_numPointsPerDirHost.resize(domainDim, numPatches);
    m_numPointsPerDir.copyToHost(m_numPointsPerDirHost.data());
    std::cout << "Distributed points per direction per patch:\n" 
              << m_numPointsPerDirHost << std::endl;
    int numPoints = 0;
    for (int p = 0; p < numPatches; p++)
        numPoints += m_numPointsPerDirHost.col(p).prod();
    m_geoPointsDeviceArray.resize(numPoints * geometry.targetDim());
    m_pointGrid.resize(numPoints * domainDim);
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
        evalAtDistributedPointsKernel, 0, numPoints);
    gridSize = (numPoints + blockSize - 1) / blockSize;
    evalAtDistributedPointsKernel<<<gridSize, blockSize>>>(
        geometry,
        m_numPointsPerDir.matrixView(domainDim, numPatches),
        m_pointGrid.matrixView(domainDim, numPoints),
        m_geoPointsDeviceArray.matrixView(geometry.targetDim(), numPoints));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in evalAtDistributedPointsKernel: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Synchronization Error in evalAtDistributedPointsKernel: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    m_geoPointsHost.resize(geometry.targetDim(), numPoints);
    //m_pointGrid.copyToHost(pointsHost.data());
    //std::cout << "Distributed parameter points:\n" 
    //          << pointsHost << std::endl;
    m_geoPointsDeviceArray.copyToHost(m_geoPointsHost.data());
    std::cout << "Evaluated points at distributed locations:\n" 
              << m_geoPointsHost << std::endl;

    if (outputMesh)
    {
        m_meshPointPatchOffsets.reserve(numPatches + 1);
        m_meshPointPatchOffsets.clear();
        m_meshEdgesPatchOffsets.reserve(numPatches + 1);
        m_meshEdgesPatchOffsets.clear();
        m_meshPointPatchOffsets.push_back(0);
        m_meshEdgesPatchOffsets.push_back(0);
        std::vector<std::vector<double>> breaks;
        breaks.reserve(numPatches * domainDim);
        int numMeshCorners = 0;
        for (int i = 0; i < numPatches; i++)
        {

            int totalNumMeshCornersInPatch = 1;
            std::vector<int> nv(domainDim);
            for (int d = 0; d < domainDim; d++)
            {
                int numEleInDir = m_assembler.basisHost().basis(i).getNumElements(d);
                nv[d] = numEleInDir + 1;
                totalNumMeshCornersInPatch *= nv[d];

                breaks.push_back(m_assembler.basisHost().basis(i).breaks(d));
            }

            int totalNumMeshlinesInPatch = 0;
            for (int d = 0; d < domainDim; ++d)
            {
                int prodOther = 1;
                for (int j = 0; j < domainDim; ++j)
                {
                    if (j == d) continue;
                    prodOther *= nv[j];
                }
                totalNumMeshlinesInPatch += (nv[d] - 1) * prodOther;
            }

            int totalNumMeshPointsInPatch  = totalNumMeshCornersInPatch + totalNumMeshlinesInPatch * numMidPoints;
            m_meshPointPatchOffsets.push_back(m_meshPointPatchOffsets.back() + totalNumMeshPointsInPatch);
            int totalNumMeshEdgesInPatch = totalNumMeshlinesInPatch * (numMidPoints + 1);
            m_meshEdgesPatchOffsets.push_back(m_meshEdgesPatchOffsets.back() + totalNumMeshEdgesInPatch);

            numMeshCorners += totalNumMeshCornersInPatch;
        }

        DeviceArray<int> meshPointPatchOffsetsDeviceData(m_meshPointPatchOffsets);
        DeviceArray<int> meshEdgesPatchOffsetsDeviceData(m_meshEdgesPatchOffsets);

        m_meshPointGrid.resize(m_meshPointPatchOffsets.back() * domainDim);
        m_meshGeoPoints.resize(m_meshPointPatchOffsets.back() * geometry.targetDim());
        m_meshEdges.resize(m_meshEdgesPatchOffsets.back() * 2); //edge connectivity
        DeviceNestedArray<double> breaksData(breaks);

        DeviceArray<int> pointCounters(numPatches), edgeCounters(numPatches);
        int minGrid, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, 
            computeMeshKernel, 0, numMeshCorners);
         int gridSize = (numMeshCorners + blockSize - 1) / blockSize;
        computeMeshKernel<<<gridSize, blockSize>>>(
            breaksData.view(),
            geometry,
            meshPointPatchOffsetsDeviceData.vectorView(),
            meshEdgesPatchOffsetsDeviceData.vectorView(),
            m_meshPointGrid.matrixView(domainDim, m_meshPointPatchOffsets.back()),
            m_meshGeoPoints.matrixView(geometry.targetDim(), m_meshPointPatchOffsets.back()),
            m_meshEdges.matrixView(2, m_meshEdgesPatchOffsets.back()),
            pointCounters.vectorView(), edgeCounters.vectorView(),
            numMidPoints, numMeshCorners);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Error in computeMeshKernel: " 
                      << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Synchronization Error in computeMeshKernel: " 
                      << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        m_meshGeoPointsHost.resize(geometry.targetDim(), m_meshPointPatchOffsets.back());
        m_meshEdgesHost.resize(2, m_meshEdgesPatchOffsets.back());
        m_meshGeoPoints.copyToHost(m_meshGeoPointsHost.data());
        m_meshEdges.copyToHost(m_meshEdgesHost.data());
        //std::cout << "m_meshPointGrid:\n" << m_meshGeoPointsHost << std::endl;
        //std::cout << "m_meshEdges:\n" << m_meshEdgesHost << std::endl;
    }
}

void GPUPostProcessor::evalFunctionsAtPoints(std::map<std::string, Eigen::MatrixXd> &data) const
{
    data.clear();
    for (std::map<std::string, GPUFunction*>::const_iterator it = m_functions.begin(); 
         it != m_functions.end(); ++it)
    {
        int domainDim = m_assembler.domainDim();
        int numPoints = m_pointGrid.size() / domainDim;
        int numPatches = m_assembler.numPatches();
        DeviceArray<double> deviceData(m_geoPointsDeviceArray.size());
        std::vector<int> numPointsPerPatchHost;
        numPointsPerPatchHost.reserve(numPatches);
        for (int p = 0; p < numPatches; p++)
        {
            int pointsInPatch = 1;
            for (int d = 0; d < domainDim; d++)
                pointsInPatch *= m_numPointsPerDirHost(d, p);
            numPointsPerPatchHost.push_back(pointsInPatch);
        }
        DeviceArray<int> numPointsPerPatchDeviceData(numPointsPerPatchHost);
        it->second->eval_into(m_pointGrid.matrixView(domainDim, numPoints), 
                              numPointsPerPatchDeviceData.vectorView(),
                              deviceData.matrixView(domainDim, numPoints));
        data[it->first].resize(domainDim, numPoints);
        deviceData.copyToHost(data[it->first].data());
        //std::cout << data[it->first] << std::endl;
        if ( data[it->first].rows() == 2 )
        {
            data[it->first].conservativeResize(3, data[it->first].cols());
            data[it->first].row(2).setZero();
        }
    }
}

void GPUPostProcessor::evalFunctionsAtMeshPoints(std::map<std::string, Eigen::MatrixXd> &data) const
{
    data.clear();
    for (std::map<std::string, GPUFunction*>::const_iterator it = m_functions.begin(); 
         it != m_functions.end(); ++it)
    {
        int domainDim = m_assembler.domainDim();
        int numPoints = m_meshPointGrid.size() / domainDim;
        int numPatches = m_assembler.numPatches();
        DeviceArray<double> deviceData(m_meshGeoPoints.size());
        std::vector<int> numPointsPerPatchHost;
        numPointsPerPatchHost.reserve(numPatches);
        for (int p = 0; p < numPatches; p++)
        {
            numPointsPerPatchHost.push_back(m_meshPointPatchOffsets[p+1] - m_meshPointPatchOffsets[p]);
        }
        DeviceArray<int> numPointsPerPatchDeviceData(numPointsPerPatchHost);
        it->second->eval_into(m_meshPointGrid.matrixView(domainDim, numPoints), 
                              numPointsPerPatchDeviceData.vectorView(),
                              deviceData.matrixView(domainDim, numPoints));
        data[it->first].resize(domainDim, numPoints);
        deviceData.copyToHost(data[it->first].data());
        //std::cout << data[it->first].transpose() << std::endl;
        if ( data[it->first].rows() == 2 )
        {
            data[it->first].conservativeResize(3, data[it->first].cols());
            data[it->first].row(2).setZero();
        }
    }
}

void GPUPostProcessor::outputToParaview(const std::string &fn, 
                                        int step, ParaviewCollection &collection) const
{
    std::map<std::string, Eigen::MatrixXd> data;
	//auto start = std::chrono::high_resolution_clock::now();
    evalFunctionsAtPoints(data);
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = end - start;
    //std::cout << "Evaluated functions at points in " << elapsed.count() << " s." << std::endl;
    //start = std::chrono::high_resolution_clock::now();
    writeParaview(fn, data, step, collection);
    //end = std::chrono::high_resolution_clock::now();
    //elapsed = end - start;
    //std::cout << "Wrote Paraview files in " << elapsed.count() << " s." << std::endl;
    if (outputMesh)
    {
        evalFunctionsAtMeshPoints(data);
        writeParaviewMesh(fn, data, step, collection);
    }
    

}

void GPUPostProcessor::writeParaview(const std::string &fn, 
    const std::map<std::string, Eigen::MatrixXd> &data, int step, 
    ParaviewCollection &collection) const
{
    std::string fileName = fn.substr(fn.find_last_of("/\\")+1);
    int dataStart = 0;
    for (int p = 0; p < m_assembler.numPatches(); p++)
    {
        int d = m_assembler.geometryHost().patch(p).getBasisDim();
        int n = m_assembler.geometryHost().patch(p).getCPDim();
        Eigen::VectorXi np = m_numPointsPerDirHost.col(p);
        int totalNumPointsPatch = np.prod();
        Eigen::MatrixXd pointsPatch = m_geoPointsHost.middleCols(dataStart, totalNumPointsPatch);
        std::map<std::string, Eigen::MatrixXd> dataPatch;
        for (const auto& item : data)
        {
            dataPatch[item.first] = 
                item.second.middleCols(dataStart, totalNumPointsPatch);
        }
        if (3 -d > 0)
        {
            np.conservativeResize(3);
            np.bottomRows(3-d).setOnes();
        }
        else if (d > 3)
        {
            std::cout << "Cannot plot 4D data.\n";
            return;
        }

        if ( 3 - n > 0 )
        {
            pointsPatch.conservativeResize(3, pointsPatch.cols());
            pointsPatch.bottomRows(3 - n).setZero();
        }
        else if (n > 3)
        {
            std::cout << "Cannot plot data with dimension higher than 3.\n";
        }
        writeParaviewSinglePatch(fn + std::to_string(step) + "_" + std::to_string(p), np, pointsPatch, dataPatch);
        collection.addTimestep(fileName + std::to_string(step), p, step, ".vts");
        dataStart += totalNumPointsPatch;
    }
}

void GPUPostProcessor::writeParaviewMesh(const std::string &fn, 
    const std::map<std::string, Eigen::MatrixXd> &data, int step, 
    ParaviewCollection &collection) const
{
    std::string fileName = fn.substr(fn.find_last_of("/\\")+1);
    for (int p = 0; p < m_assembler.numPatches(); p++)
    {
        Eigen::MatrixXd pointsPatch = m_meshGeoPointsHost.middleCols(m_meshPointPatchOffsets[p], 
            m_meshPointPatchOffsets[p+1] - m_meshPointPatchOffsets[p]);
        //std::cout << "Mesh points patch " << p << ":\n" << pointsPatch.transpose() << std::endl;
        Eigen::MatrixXi edgesPatch = m_meshEdgesHost.middleCols(m_meshEdgesPatchOffsets[p],
            m_meshEdgesPatchOffsets[p+1] - m_meshEdgesPatchOffsets[p]);
        //std::cout << "Mesh edges patch " << p << ":\n" << edgesPatch.transpose() << std::endl;
        std::map<std::string, Eigen::MatrixXd> dataPatch;
        for (const auto& item : data)
        {
            dataPatch[item.first] = 
                item.second.middleCols(m_meshPointPatchOffsets[p], 
                    m_meshPointPatchOffsets[p+1] - m_meshPointPatchOffsets[p]);
            //std::cout << "Data " << item.first << " patch " << p << ":\n" << dataPatch[item.first].transpose() << std::endl;
        }
        writeParaviewSinglePatchMesh(fn + std::to_string(step) + "_" + std::to_string(p), pointsPatch, edgesPatch, dataPatch);
        collection.addTimestep(fileName + std::to_string(step), p, step, ".vtp");
    }
}

void GPUPostProcessor::writeParaviewSinglePatch(const std::string &fn, 
        const Eigen::VectorXi &np, 
        const Eigen::MatrixXd &points,
        const std::map<std::string, Eigen::MatrixXd> &data) const
{
    const int n = points.rows();

    std::string mfn(fn);
    mfn.append(".vts");
    std::ofstream file(mfn.c_str());
    file << std::fixed; // no exponents
    file << std::setprecision (11);

    //std::cout << np << std::endl;

    file <<"<?xml version=\"1.0\"?>\n";
    file <<"<VTKFile type=\"StructuredGrid\" version=\"0.1\">\n";
    file <<"<StructuredGrid WholeExtent=\"0 "<< np(0)-1<<" 0 "<<np(1)-1<<" 0 "
         << (np.size()>2 ? np(2)-1 : 0) <<"\">\n";
    file <<"<Piece Extent=\"0 "<< np(0)-1<<" 0 "<<np(1)-1<<" 0 "
         << (np.size()>2 ? np(2)-1 : 0) <<"\">\n";

    file <<"<PointData>\n";
    for (const auto& item : data)
    {
        file << "<DataArray type=\"Float32\" Name=\"" << item.first << "\" format=\"ascii\" NumberOfComponents=\"" << (item.second.rows() ==1 ? 1 : 3) << "\">\n";
        if (item.second.rows() == 1)
            for (int i = 0; i < item.second.cols(); ++i)
                file << item.second(0, i) << " ";
        else
        {
            for (int j = 0; j < item.second.cols(); ++j)
            {
                for (int i = 0; i < item.second.rows(); ++i)
                    file << item.second(i, j) << " ";
                for (int i = item.second.rows(); i < 3; ++i)
                    file << "0 ";
                file << "\n";
            }
        }
        file << "</DataArray>\n";
    }
    file <<"</PointData>\n";
    file <<"<Points>\n";
    file <<"<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n";
    for ( int j=0; j<points.cols(); ++j)
    {
        for ( int i=0; i!=n; ++i)
            file<< points(i,j) <<" ";
        for ( int i=n; i<3; ++i)
            file<<"0 ";
        file << "\n";
    }
    file <<"</DataArray>\n";
    file <<"</Points>\n";
    file <<"</Piece>\n";
    file <<"</StructuredGrid>\n";
    file <<"</VTKFile>\n";

    file.close();
}

void GPUPostProcessor::writeParaviewSinglePatchMesh(const std::string &fn, 
    const Eigen::MatrixXd &points, const Eigen::MatrixXi &edges,
    const std::map<std::string, Eigen::MatrixXd> &data) const
{
    std::string mfn(fn);
    mfn.append(".vtp");
    std::ofstream file(mfn.c_str());
    file << std::fixed;
    file << std::setprecision (11);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "<PolyData>\n";
    file << "<Piece NumberOfPoints=\"" << points.cols() << "\" NumberOfVerts=\"0\" NumberOfLines=\""
         << edges.cols() << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

    file << "<PointData>\n";
    for (const auto& item : data)
    {
        file << "<DataArray type=\"Float32\" Name=\"" << item.first << "\" format=\"ascii\" NumberOfComponents=\"" << (item.second.rows() ==1 ? 1 : 3) << "\">\n";
        if (item.second.rows() == 1)
            for (int i = 0; i < item.second.cols(); ++i)
                file << item.second(0, i) << " ";
        else
        {
            for (int j = 0; j < item.second.cols(); ++j)
            {
                for (int i = 0; i < item.second.rows(); ++i)
                    file << item.second(i, j) << " ";
                for (int i = item.second.rows(); i < 3; ++i)
                    file << "0 ";
                file << "\n";
            }
        }
        file << "</DataArray>\n";
    }
    file << "</PointData>\n";

    file << "<Points>\n";
    file << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for ( int j=0; j<points.cols(); ++j)
    {
        for ( int i=0; i!=points.rows(); ++i)
            file<< points(i,j) <<" ";
        for ( int i=points.rows(); i<3; ++i)
            file<<"0 ";
        file << "\n";
    }
    file <<"</DataArray>\n";
    file <<"</Points>\n";

    file << "<Lines>\n";
    file << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int j = 0; j < edges.cols(); ++j)
    {
        for (int i = 0; i < edges.rows(); ++i)
            file << edges(i, j) << " ";
        file << "\n";
    }
    file << "</DataArray>\n";
    file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    int count = 0;
    for (int j = 0; j < edges.cols(); ++j)
    {
        count += 2;
        file << count << " ";
    }
    file << "\n";
    file << "</DataArray>\n";
    file << "</Lines>\n";

    file <<"</Piece>\n";
    file <<"</PolyData>\n";
    file <<"</VTKFile>\n";

    file.close();
}
