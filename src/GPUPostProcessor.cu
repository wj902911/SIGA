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
                double geoJacobianData[2] = {0.0}; //max 3D
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
            break;
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
void testKernel()
{

}

GPUPostProcessor::GPUPostProcessor(const GPUAssembler &assembler, 
                                   const std::vector<int>& numPointsPerPatch)
    : m_assembler(assembler)
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
    //std::cout << "Distributed points per direction per patch:\n" 
    //          << m_numPointsPerDirHost << std::endl;
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
    //std::cout << "Evaluated points at distributed locations:\n" 
    //          << pointsHost << std::endl;
}

void GPUPostProcessor::evalFunctionsAtPoints(std::map<std::string, Eigen::MatrixXd> &data) const
{
    for (std::map<std::string, GPUFunction*>::const_iterator it = m_functions.begin(); 
         it != m_functions.end(); ++it)
    {
        int domainDim = m_assembler.domainDim();
        int numPoints = m_pointGrid.size() / domainDim;
        int numPatches = m_assembler.numPatches();
        DeviceArray<double> deviceData(m_geoPointsDeviceArray.size());
        it->second->eval_into(m_pointGrid.matrixView(domainDim, numPoints), 
                              m_numPointsPerDir.matrixView(domainDim, numPatches),
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
}

void GPUPostProcessor::writeParaview(const std::string &fn, 
    const std::map<std::string, Eigen::MatrixXd> &data, int step, 
    ParaviewCollection &collection) const
{
    std::string fileName = fn.substr(fn.find_last_of("/\\")+1);
    MultiPatchDeviceView geometry = m_assembler.geometryView();
    for (int p = 0; p < m_assembler.numPatches(); p++)
    {
        int d = m_assembler.geometryHost().patch(p).getBasisDim();
        int n = m_assembler.geometryHost().patch(p).getCPDim();
        Eigen::VectorXi np = m_numPointsPerDirHost.col(p);
        int totalNumPointsPatch = np.prod();
        Eigen::MatrixXd pointsPatch = m_geoPointsHost.middleCols(p * totalNumPointsPatch, totalNumPointsPatch);
        std::map<std::string, Eigen::MatrixXd> dataPatch;
        for (const auto& item : data)
        {
            dataPatch[item.first] = 
                item.second.middleCols(p * totalNumPointsPatch, totalNumPointsPatch);
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
    }
    file <<"</DataArray>\n";
    file <<"</Points>\n";
    file <<"</Piece>\n";
    file <<"</StructuredGrid>\n";
    file <<"</VTKFile>\n";

    file.close();
}
