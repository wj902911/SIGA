#pragma once

#include <cstdio>
#include <TensorBsplineBasisDeviceView.h>

class PatchDeviceView
{
private:
    int m_domainDim;
    int m_targetDim;
    TensorBsplineBasisDeviceView m_basis;
    DeviceMatrixView<double> m_controlPoints;

public:
    __host__ __device__
    PatchDeviceView(int domainDim, int targetDim,
                    TensorBsplineBasisDeviceView basis,
                    DeviceMatrixView<double> controlPoints)
                  : m_domainDim(domainDim), m_targetDim(targetDim), 
                    m_basis(basis), m_controlPoints(controlPoints)
    {
    }

    __host__ __device__
    PatchDeviceView(int domainDim, int targetDim,
                    DeviceVectorView<int> knotsOffset,
                    DeviceVectorView<int> knotsOrders,
                    DeviceVectorView<double> knotsPool,
                    DeviceMatrixView<double> controlPoints)
                  : m_domainDim(domainDim), m_targetDim(targetDim),
                    m_basis(domainDim, knotsOffset, knotsOrders, knotsPool), 
                    m_controlPoints(controlPoints)
    {
    }

    __device__
    PatchDeviceView(int domainDim, int targetDim,
                    DeviceVectorView<int> intData,
                    DeviceVectorView<double> knotsPool,
                    DeviceMatrixView<double> controlPoints)
                  : m_domainDim(domainDim), m_targetDim(targetDim),
                    m_basis(domainDim, 
                            DeviceVectorView<int>(intData.data(), 2 * domainDim + 1),
                            knotsPool), 
                    m_controlPoints(controlPoints)
    {
    }

    __device__
    int domainDim() const { return m_basis.dim(); }

    __device__
    int targetDim() const { return m_controlPoints.cols(); }

    __device__
    TensorBsplineBasisDeviceView basis() const { return m_basis; }

    __device__
    void print() const
    {
            printf("Knot vectors:\n");
            for (int d = 0; d < m_basis.dim(); d++)
            {
                KnotVectorDeviceView kv = m_basis.knotVector(d);
                printf("Direction %d:\nOrder %d\nKnots:\n", d, kv.order());
                kv.knots().print();
            }
            printf("Control points:\n");
            m_controlPoints.print();
    }
    
};