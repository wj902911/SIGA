#pragma once

#include <ParaviewCollection.h>
//#include <MultiPatchDeviceView.h>
#include <GPUAssembler.h>
#include <GPUFunction.h>
#include <map>
#include <string>

class GPUPostProcessor
{
private:
    //MultiPatchDeviceView m_geometryDeviceView;
    const GPUAssembler& m_assembler;
    std::map<std::string, GPUFunction*> m_functions;
    DeviceArray<int> m_numPointsPerDir;
    DeviceArray<double> m_pointGrid;
    DeviceArray<double> m_geoPointsDeviceArray;
    Eigen::MatrixXi m_numPointsPerDirHost;
    Eigen::MatrixXd m_geoPointsHost;

public:
    GPUPostProcessor(const GPUAssembler &assembler, 
                     const std::vector<int>& numPointsPerPatch);

    void addFunction(const std::string &name, GPUFunction* function)
    { m_functions[name] = function; }

    void evalFunctionsAtPoints(std::map<std::string, Eigen::MatrixXd>& data) const;

    void outputToParaview(const std::string &fn, 
                          int step, 
                          ParaviewCollection& collection) const;

    void writeParaview(const std::string &fn, 
                       const std::map<std::string, 
                       Eigen::MatrixXd>& data,
                       int step,
                       ParaviewCollection& collection) const;
    
    void writeParaviewSinglePatch(const std::string &fn, 
                                  const Eigen::VectorXi &np, 
                                  const Eigen::MatrixXd &points,
                                  const std::map<std::string, Eigen::MatrixXd>& data) const;
    
};