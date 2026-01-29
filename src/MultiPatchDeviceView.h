#pragma once

#include <PatchDeviceView.h>
#include <MultiBasisDeviceView.h>

class MultiPatchDeviceView
{
private:
    int m_numPatches;

    int m_domainDim;
    int m_targetDim;

    DeviceVectorView<int> m_patchIntDataOffsets;
    DeviceVectorView<int> m_patchKnotsPoolOffsets;
    DeviceVectorView<int> m_patchControlPointsPoolOffsets;
    DeviceVectorView<int> m_intData;;
    DeviceVectorView<double> m_knotsPools;
    DeviceVectorView<double> m_controlPointsPools;

    //DeviceVectorView<int> m_knotsOffset;
    //DeviceVectorView<int> m_knotsOrders;
    //DeviceVectorView<double> m_knotsPool;

    //DeviceVectorView<int> m_controlPointsOffset;
    //DeviceVectorView<double> m_controlPointsPool;

public:
    __host__ __device__
    MultiPatchDeviceView(int numPatches,
                         int domainDim,
                         int targetDim,
                         DeviceVectorView<int> patchIntDataOffsets,
                         DeviceVectorView<int> patchKnotsPoolOffsets,
                         DeviceVectorView<int> patchControlPointsPoolOffsets,
                         DeviceVectorView<int> intData,
                         DeviceVectorView<double> knotsPools,
                         DeviceVectorView<double> controlPointsPools)
                       : m_numPatches(numPatches),
                         m_domainDim(domainDim),
                         m_targetDim(targetDim),
                         m_patchIntDataOffsets(patchIntDataOffsets),
                         m_patchKnotsPoolOffsets(patchKnotsPoolOffsets),
                         m_patchControlPointsPoolOffsets(patchControlPointsPoolOffsets),
                         m_intData(intData),
                         m_knotsPools(knotsPools),
                         m_controlPointsPools(controlPointsPools)
    {
    }

    __host__ __device__
    MultiPatchDeviceView(int targetDim,
                         MultiBasisDeviceView multiBasis,
                         DeviceVectorView<int> patchControlPointsPoolOffsets,
                         DeviceVectorView<double> controlPointsPools)
                       : m_numPatches(multiBasis.numPatches()),
                         m_domainDim(multiBasis.domainDim()),
                         m_targetDim(targetDim),
                         m_patchIntDataOffsets(multiBasis.patchIntDataOffsets()),
                         m_patchKnotsPoolOffsets(multiBasis.patchKnotsPoolOffsets()),
                         m_patchControlPointsPoolOffsets(patchControlPointsPoolOffsets),
                         m_intData(multiBasis.intData()),
                         m_knotsPools(multiBasis.knotsPools()),
                         m_controlPointsPools(controlPointsPools)
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

        return TensorBsplineBasisDeviceView(m_domainDim,
                                            patchIntData,
                                            patchKnotsPool);
    }

    __device__
    DeviceMatrixView<double> controlPoints(int patchIdx) const
    {
        int controlPointsPoolOffsetStart = m_patchControlPointsPoolOffsets[patchIdx];
        int controlPointsPoolOffsetEnd = m_patchControlPointsPoolOffsets[patchIdx + 1];

        DeviceVectorView<double> patchControlPointsPool(m_controlPointsPools.data() + controlPointsPoolOffsetStart,
                                                        controlPointsPoolOffsetEnd - controlPointsPoolOffsetStart);

        int numControlPoints = (controlPointsPoolOffsetEnd - controlPointsPoolOffsetStart) / m_targetDim;

        return DeviceMatrixView<double>(patchControlPointsPool.data(),
                                        numControlPoints,
                                        m_targetDim);
    }

    __device__
    PatchDeviceView patch(int patchIdx) const
    {
        return PatchDeviceView(m_domainDim,
                               m_targetDim,
                               basis(patchIdx),
                               controlPoints(patchIdx));
    }

#if 0
    __device__
    DeviceVectorView<int> knotOffsets() const
    {
        return DeviceVectorView<int>(m_knotsOffset, m_numPatches * m_domainDim + 1);
    }
#endif

    __device__
    int* intDataPtr() const
    {
        return m_intData.data();
    }

    __device__
    DeviceVectorView<double> knotsPools() const
    {
        return m_knotsPools;
    }

    __device__
    void print() const
    {
        for (int p = 0; p < m_numPatches; p++)
        {
            printf("Patch %d:\n", p);
            patch(p).print();
        }
    }
};