#pragma once

#include <TensorBsplineBasisDeviceView.h>

class MultiBasisDeviceView
{
private:
    int m_numPatches;
    int m_domainDim;
    DeviceVectorView<int> m_patchIntDataOffsets;
    DeviceVectorView<int> m_patchKnotsPoolOffsets;
    DeviceVectorView<int> m_intData;;
    DeviceVectorView<double> m_knotsPools;

public:
    __host__ __device__
    MultiBasisDeviceView(int numPatches,
                         int domainDim,
                         DeviceVectorView<int> patchIntDataOffsets,
                         DeviceVectorView<int> patchKnotsPoolOffsets,
                         DeviceVectorView<int> intData,
                         DeviceVectorView<double> knotsPools)
                       : m_numPatches(numPatches),
                         m_domainDim(domainDim),
                         m_patchIntDataOffsets(patchIntDataOffsets),
                         m_patchKnotsPoolOffsets(patchKnotsPoolOffsets),
                         m_intData(intData),
                         m_knotsPools(knotsPools)
    {
    }

    __device__
    TensorBsplineBasisDeviceView basis(int patchIdx) const
    {
        int intDataOffsetStart = m_patchIntDataOffsets[patchIdx];
        int intDataOffsetEnd = m_patchIntDataOffsets[patchIdx + 1];

        DeviceVectorView<int> patchIntData(m_intData.data() + intDataOffsetStart,
                                           intDataOffsetEnd - intDataOffsetStart);

        int knotsPoolOffsetStart = m_patchKnotsPoolOffsets[patchIdx];
        int knotsPoolOffsetEnd = m_patchKnotsPoolOffsets[patchIdx + 1];

        DeviceVectorView<double> patchKnotsPool(m_knotsPools.data() + knotsPoolOffsetStart,
                                                knotsPoolOffsetEnd - knotsPoolOffsetStart);

        return TensorBsplineBasisDeviceView(m_domainDim, patchIntData, patchKnotsPool);
    }

    __host__ __device__
    int numPatches() const { return m_numPatches; }

    __host__ __device__
    int domainDim() const { return m_domainDim; }

    __host__ __device__
    DeviceVectorView<int> patchIntDataOffsets() const { return m_patchIntDataOffsets; }

    __host__ __device__
    DeviceVectorView<int> patchKnotsPoolOffsets() const { return m_patchKnotsPoolOffsets; }

    __host__ __device__
    DeviceVectorView<int> intData() const { return m_intData; }

    __host__ __device__
    DeviceVectorView<double> knotsPools() const { return m_knotsPools; }

    __device__
    void print() const
    {
        for (int p = 0; p < m_numPatches; p++)
        {
            printf("Basis for Patch %d:\n", p);
            basis(p).print();
        }
    }
};