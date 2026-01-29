#pragma once

#include <DeviceArray.h>
#include <MultiBasis.h>
#include <MultiBasisDeviceView.h>

class MultiBasisDeviceData
{
private:
    int m_numPatches;
    int m_domainDim;
    DeviceArray<int> m_intData;
    DeviceArray<double> m_knotsPools;

public:
    __host__
    MultiBasisDeviceData(int numPatches,
                         int domainDim,
                         const std::vector<int>& intData,
                         const std::vector<double>& knotsPools)
                       : m_numPatches(numPatches),
                         m_domainDim(domainDim),
                         m_intData(intData),
                         m_knotsPools(knotsPools)
    {
    }

    __host__
    MultiBasisDeviceData(const MultiBasis& multibasis)
    {
        m_numPatches = multibasis.getNumBases();
        m_domainDim = multibasis.getDim();
        multibasis.getData(m_intData, m_knotsPools);
    }

    __host__
    MultiBasisDeviceView deviceView() const
    {
        int patchIntDataOffsetsSize = m_numPatches + 1;
        int* dataStart = m_intData.data();
        DeviceVectorView<int> patchIntDataOffsets(dataStart, patchIntDataOffsetsSize);
        dataStart += patchIntDataOffsetsSize;
        DeviceVectorView<int> patchKnotsPoolOffsets(dataStart, patchIntDataOffsetsSize);
        dataStart += patchIntDataOffsetsSize;
        DeviceVectorView<int> intData(dataStart,
                                      m_intData.size() - (patchIntDataOffsetsSize * 2));
        DeviceVectorView<double> knotsPools(m_knotsPools.data(), m_knotsPools.size());

        return MultiBasisDeviceView(m_numPatches,
                                    m_domainDim,
                                    patchIntDataOffsets,
                                    patchKnotsPoolOffsets,
                                    intData,
                                    knotsPools);
    }
};