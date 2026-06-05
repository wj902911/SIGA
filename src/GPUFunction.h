#pragma once

#include <MultiPatchDeviceData.h>
#include <Eigen/Core>

class GPUFunction
{
private:
    const MultiPatch &m_multiPatchHost;
    MultiPatchDeviceData m_multiPatchDeviceData;
public:
    GPUFunction(const MultiPatch &multiPatchHost);

    Eigen::MatrixXd eval(int patch, const Eigen::MatrixXd &u) const;

    void eval_into(DeviceMatrixView<double> gridPoints,
                   DeviceVectorView<int> numPointsPerPatch,
                   DeviceMatrixView<double> values) const;

    int domainDim() const
    { return m_multiPatchHost.getBasisDim(); }

    int targetDim() const
    { return m_multiPatchHost.getCPDim(); }

    int numPatches() const
    { return m_multiPatchHost.getNumPatches(); }
    
    MultiPatchDeviceView multiPatchDeviceView() const
    { return m_multiPatchDeviceData.deviceView(); }
};
