#include "Postprocessor.h"
#include <DeviceObjectPointer.h>
#include <fstream>
#include <iomanip>

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

void PostProcessor::evalFunctionsAtPoints(std::map<std::string, Eigen::MatrixXd> &data, 
                                          const Eigen::MatrixXi &numPointsPerDir) const
{
    for (std::map<std::string, Function*>::const_iterator it = m_functions.begin(); it != m_functions.end(); ++it)
    {
        it->second->eval_into(numPointsPerDir, data[it->first]);
        //std::cout << data[it->first] << std::endl;
        if ( data[it->first].rows() == 2 )
        {
            data[it->first].conservativeResize(3, data[it->first].cols());
            data[it->first].row(2).setZero();
        }
    }
}

void PostProcessor::outputToParaview(const std::string &fn, 
                                     const Eigen::VectorXi& numPoints, 
                                     int step, 
                                     ParaviewCollection& collection) const
{
    Eigen::MatrixXi numPointsPerDir;
    distributePoints(numPoints, numPointsPerDir);
    //std::cout << "Number of points per direction per patch:\n" << numPointsPerDir << std::endl;
    Eigen::MatrixXd points;
    evalGeometryAtPoints(numPointsPerDir, points);
    std::map<std::string, Eigen::MatrixXd> data;
    evalFunctionsAtPoints(data, numPointsPerDir);
    writeParaview(fn, numPointsPerDir, points, data, step, collection);
}

void PostProcessor::writeParaview(const std::string& fn, 
                                  const Eigen::MatrixXi& numPointsPerDir, 
                                  const Eigen::MatrixXd& points,
                                  const std::map<std::string, Eigen::MatrixXd>& data,
                                  int step,
                                  ParaviewCollection& collection) const
{
    std::string fileName = fn.substr(fn.find_last_of("/\\")+1);
    for (int p = 0; p < m_geometry.getNumPatches(); p++)
    {
        int d = m_geometry.patch(p).getBasisDim();
        int n = m_geometry.patch(p).getCPDim();
        Eigen::VectorXi np = numPointsPerDir.col(p);
        int totalNumPointsPatch = np.prod();
        Eigen::MatrixXd pointsPatch = points.middleCols(p * totalNumPointsPatch, totalNumPointsPatch);
        std::map<std::string, Eigen::MatrixXd> dataPatch;
        for (const auto& item : data)
        {
            dataPatch[item.first] = item.second.middleCols(p * totalNumPointsPatch, totalNumPointsPatch);
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

void PostProcessor::writeParaviewSinglePatch(const std::string &fn, 
                                             const Eigen::VectorXi &np, 
                                             const Eigen::MatrixXd &points,
                                             const std::map<std::string, Eigen::MatrixXd>& data) const
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
