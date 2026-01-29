#pragma once

#include <MultiPatch.h>
#include <MultiPatchDeviceView.h>
#include <DeviceArray.h>

class MultiPatchDeviceData
{
private:
    int m_numPatches;
    int m_domainDim;
    int m_targetDim;
    //std::vector<int> m_intDataOffsets;
    //std::vector<int> m_doubleDataOffsets;
    DeviceArray<int> m_intData;
    DeviceArray<double> m_knotsPools;
    DeviceArray<int> m_patchControlPointsPoolOffsets;
    DeviceArray<double> m_controlPointsPools;

public:
    __host__
    MultiPatchDeviceData(int numPatches,
                         int domainDim,
                         int targetDim,
                         //const std::vector<int>& intDataOffsets,
                         //const std::vector<int>& doubleDataOffsets,
                         const std::vector<int>& intData,
                         const std::vector<double>& knotsPools,
                         const std::vector<int>& patchControlPointsPoolOffsets,
                         const std::vector<double>& controlPointsPools)
                       : m_numPatches(numPatches),
                         m_domainDim(domainDim),
                         m_targetDim(targetDim),
                         //m_intDataOffsets(intDataOffsets),
                         //m_doubleDataOffsets(doubleDataOffsets),
                         m_intData(intData),
                         m_knotsPools(knotsPools),
                         m_patchControlPointsPoolOffsets(patchControlPointsPoolOffsets),
                         m_controlPointsPools(controlPointsPools)
    {
    }

    __host__
    MultiPatchDeviceData(const MultiPatch& multipatch)
                        :m_numPatches(multipatch.getNumPatches()),
                         m_domainDim(multipatch.getBasisDim()),
                         m_targetDim(multipatch.getCPDim())
    {
        multipatch.getData(m_intData,
                           m_knotsPools,
                           m_patchControlPointsPoolOffsets,
                           m_controlPointsPools);
    }

#if 0
    __host__
    MultiPatchDeviceData(const MultiPatch& multiPatch);
#endif

    __host__
    MultiPatchDeviceView deviceView() const;
};