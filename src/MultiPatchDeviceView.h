#pragma once

#include <PatchDeviceView.h>
#include <MultiBasisDeviceView.h>
#include <GaussPointsDeviceView.h>

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
    int* intDataPtr() const { return m_intData.data(); }

    __device__
    DeviceVectorView<double> knotsPools() const { return m_knotsPools; }

    __device__
    void print() const
    {
        for (int p = 0; p < m_numPatches; p++)
        {
            printf("Patch %d:\n", p);
            patch(p).print();
        }
    }

    __host__ __device__
    int numPatches() const { return m_numPatches; }

    __host__ __device__
    int domainDim() const { return m_domainDim; }

    __host__ __device__
    int targetDim() const { return m_targetDim; }

    __device__
    int numControlPoints(int patchIdx) const
    { return patch(patchIdx).numControlPoints(); }

    __device__
    int totalNumControlPoints() const
    {
        int total = 0;
        for (int i = 0; i < m_numPatches; i++)
        {
            total += numControlPoints(i);
        }
        return total;
    }

    __device__
    int threadPatchAndDof(int idx, int& p, int& unk) const
    {
        int dim = targetDim();
        int point_idx_patch = idx;
        int point_idx_dof = idx;
        int total_points = totalNumControlPoints();
        for (int d = 0; d < dim; d++)
        {
            if (point_idx_dof < total_points)
            {
                unk = d;
                break;
            }
            point_idx_dof -= total_points;
            point_idx_patch = point_idx_dof;
        }
        for (int i = 0; i < numPatches(); i++)
        {
            int patch_points = numControlPoints(i);
            if (point_idx_patch < patch_points) 
            {
            	p = i;
            	break;
            }
            point_idx_patch -= patch_points;
        }
        return point_idx_patch;
    }

    __device__
    void setCoefficients(int patchIndex, int row, int col, double value)
    { patch(patchIndex).setCoefficients(row, col, value); }

    __device__
    int threadPatch(int idx, int& patch_idx) const
    {
        int point_idx = idx;
        for (int i = 0; i < m_numPatches; i++)
        {
            int patch_points = patch(i).basis().totalNumGPs();
            if (point_idx < patch_points) 
            {
                patch_idx = i;
                break;
            }
            point_idx -= patch_points;
        }
        return point_idx;
    }

    __device__
    double gsPoint(int idx, int patch_idx, GaussPointsDeviceView gps,
                   DeviceVectorView<double> result) const
    { return patch(patch_idx).basis().gsPoint(idx, gps, result); }
};