#pragma once

#include <GaussPointsDeviceView.h>

class MultiGaussPointsDeviceView
{
private:
    int m_dim = 0;
    DeviceVectorView<int> m_patchOffsetsOffsets;
    DeviceVectorView<int> m_patchGaussPointsOffsets;
    DeviceVectorView<int> m_offsetsPool;
    DeviceVectorView<double> m_gaussPointsPool;
    DeviceVectorView<double> m_gaussWeightsPool;

public:
    __host__
    MultiGaussPointsDeviceView( int dim,
        DeviceVectorView<int> patchOffsetsOffsets,
        DeviceVectorView<int> patchGaussPointsOffsets,
        DeviceVectorView<int> offsetsPool,
        DeviceVectorView<double> gaussPointsPool,
        DeviceVectorView<double> gaussWeightsPool)
        : m_patchOffsetsOffsets(patchOffsetsOffsets),
          m_patchGaussPointsOffsets(patchGaussPointsOffsets),
          m_offsetsPool(offsetsPool),
          m_gaussPointsPool(gaussPointsPool),
          m_gaussWeightsPool(gaussWeightsPool),
          m_dim(dim)
    {
    }

    __device__
    GaussPointsDeviceView patchGaussPoints(int patchIdx) const
    {
        int offsetsStart = (patchIdx == 0) ? 0 : m_patchOffsetsOffsets[patchIdx - 1];
        int offsetsEnd = m_patchOffsetsOffsets[patchIdx];
        int numOffsets = offsetsEnd - offsetsStart;
        int patchGaussPointsStart = (patchIdx == 0) ? 0 : m_patchGaussPointsOffsets[patchIdx - 1];
        int patchGaussPointsEnd = m_patchGaussPointsOffsets[patchIdx];
        int numGaussPoints = patchGaussPointsEnd - patchGaussPointsStart;

        DeviceNestedArrayView<double> gaussPoints(
            m_offsetsPool.data() + offsetsStart,
            numOffsets,
            m_gaussPointsPool.data() + patchGaussPointsStart,
            numGaussPoints);

        DeviceNestedArrayView<double> gaussWeights(
            m_offsetsPool.data() + offsetsStart,
            numOffsets,
            m_gaussWeightsPool.data() + patchGaussPointsStart,
            numGaussPoints);

        return GaussPointsDeviceView(m_dim, gaussPoints, gaussWeights);
    }

    __device__
    GaussPointsDeviceView operator[](int patchIdx) const
    { return patchGaussPoints(patchIdx); }

    __device__
    void print() const
    {
        int numPatches = m_patchOffsetsOffsets.size();
        for (int p = 0; p < numPatches; ++p)
        {
            printf("Patch %d Gauss Points and Weights:\n", p);
            patchGaussPoints(p).print();
        }
    }

    __host__ __device__
    int dim() const { return m_dim; }

};