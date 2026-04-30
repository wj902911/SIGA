#pragma once

#include <DeviceArray.h>
#include <MultiBasis.h>
#include <MultiBasisDeviceView.h>

class MultiBasisDeviceData
{
private:
    int m_numPatches = 0;
    int m_domainDim = 0;
    DeviceArray<int> m_intData;
    DeviceArray<double> m_knotsPools;
    DeviceNestedArray<int> m_multSumsOffsets;
    DeviceNestedArray<int> m_multSums;

public:
	__host__
    MultiBasisDeviceData() = default;

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
    MultiBasisDeviceData(int numPatches,
                         int domainDim,
                         const std::vector<int>& intData,
                         const std::vector<double>& knotsPools,
                         const std::vector<std::vector<int>>& multSumsOffsets,
                         const std::vector<std::vector<int>>& multSums)
                       : m_numPatches(numPatches),
                         m_domainDim(domainDim),
                         m_intData(intData),
                         m_knotsPools(knotsPools),
                         m_multSumsOffsets(multSumsOffsets),
                         m_multSums(multSums)
    {
    }

    __host__
    MultiBasisDeviceData(const MultiBasis& multibasis)
    {
        m_numPatches = multibasis.getNumBases();
        m_domainDim = multibasis.getDim();
        multibasis.getData(m_intData, m_knotsPools,
                           m_multSumsOffsets, m_multSums);
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
                                    knotsPools,
                                    m_multSumsOffsets.view(),
                                    m_multSums.view());
    }
};