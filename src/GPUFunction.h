#pragma once

#include <MultiPatchDeviceData.h>

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
    const MultiPatch &m_displacementHost;
    MultiPatchDeviceData m_displacementDeviceData;
    //const MultiPatchDeviceView &m_displacementDeviceView;
public:

#if 0
    GPUDisplacementFunction(const MultiPatchDeviceView &view) 
        : m_displacementDeviceView(view) 
    {
        GPUFunctionPrintKernel<<<1, 1>>>(m_displacementDeviceView);
        cudaDeviceSynchronize();
    }
#else
    //GPUDisplacementFunction(const MultiPatchDeviceView &view);
    GPUDisplacementFunction(const MultiPatch &displacementHost);
#endif

    void eval_into(DeviceMatrixView<double> gridPoints,
                   DeviceVectorView<int> numPointsPerPatch,
                   DeviceMatrixView<double> values) const override;
    
    const MultiPatchDeviceView& displacementDeviceView() const
    { return m_displacementDeviceData.deviceView(); }

    MultiPatchDeviceView displacementDeviceView()
    { return m_displacementDeviceData.deviceView(); }
};