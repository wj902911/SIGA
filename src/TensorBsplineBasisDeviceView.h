#pragma once

#include <cstdio>
#include <KnotVectorDeviceView.h>

class TensorBsplineBasisDeviceView
{
private:
    int m_dim;
    DeviceVectorView<int> m_knotsOffset;
    DeviceVectorView<int> m_knotsOrders;
    //DeviceVectorView<int> m_intData; // combined storage for offsets and orders
    DeviceVectorView<double> m_knotsPool;

public:
    __device__
    TensorBsplineBasisDeviceView(int dim,
                                 DeviceVectorView<int> intData,
                                 DeviceVectorView<double> knotsPool)
                               : m_dim(dim),
                                 m_knotsOffset(intData.data() + dim, dim + 1),
                                 m_knotsOrders(intData.data(), dim),
                                 m_knotsPool(knotsPool)
    {
    }

    __device__
    TensorBsplineBasisDeviceView(int dim,
                                 DeviceVectorView<int> knotsOffset,
                                 DeviceVectorView<int> knotsOrders,
                                 DeviceVectorView<double> knotsPool)
                               : m_dim(dim),
                                 m_knotsOffset(knotsOffset),
                                 m_knotsOrders(knotsOrders),
                                 m_knotsPool(knotsPool)
    {
    }

    __device__
    KnotVectorDeviceView knotVector(int dir) const
    {
        return KnotVectorDeviceView(m_knotsOrders[dir],
                                    m_knotsOffset[dir + 1] - m_knotsOffset[dir],
                                    m_knotsPool.data() + m_knotsOffset[dir] - m_knotsOffset[0]);
    }

    __device__
    int dim() const { return m_dim; }

    __device__
    DeviceVectorView<int> knotsOffsets() const
    {
        return m_knotsOffset;
    }

    __device__
    DeviceVectorView<int> knotsOrders() const
    {
        return m_knotsOrders;
    }
    __device__
    DeviceVectorView<double> knotsPool() const
    {
        return m_knotsPool;
    }

    __device__
    void print() const
    {
        for (int d = 0; d < m_dim; d++)
        {
            KnotVectorDeviceView kv = knotVector(d);
            kv.print();
        }
    }

};