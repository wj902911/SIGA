#pragma once

#include <MultiPatchDeviceView.h>

class GPUFunction
{
public:
    virtual void eval_into(DeviceMatrixView<double> gridPoints,
                           DeviceVectorView<int> numPointsPerPatch,
                           DeviceMatrixView<double> values) const = 0;
};

class GPUDisplacementFunction : public GPUFunction
{
private:
    const MultiPatchDeviceView &m_displacementDeviceView;
public:

#if 0
    GPUDisplacementFunction(const MultiPatchDeviceView &view) 
        : m_displacementDeviceView(view) 
    {
        GPUFunctionPrintKernel<<<1, 1>>>(m_displacementDeviceView);
        cudaDeviceSynchronize();
    }
#else
    GPUDisplacementFunction(const MultiPatchDeviceView &view);
#endif

    void eval_into(DeviceMatrixView<double> gridPoints,
                   DeviceVectorView<int> numPointsPerPatch,
                   DeviceMatrixView<double> values) const override;
    
    const MultiPatchDeviceView& displacementDeviceView() const
    { return m_displacementDeviceView; }
};