#pragma once

#include <cstdio>
#include <DeviceVectorView.h>

class KnotVectorDeviceView
{
private:
    int m_order = 0;
    DeviceVectorView<double> m_knots;

public:
    __host__ __device__
    KnotVectorDeviceView(int order, int numKnots, double* knots)
    : m_order(order), m_knots(knots, numKnots) { }

    __device__
    DeviceVectorView<double> knots() const { return m_knots; }

    __device__
    int order() const { return m_order; }

    __device__
    void print() const
    {
        printf("Knot Vector (order %d):\n", m_order);
        m_knots.print();
    }

    __host__ __device__
    int numElements() const { return m_knots.size() - m_order * 2 - 1; }

    __host__ __device__
    int numGaussPoints() const { return m_order + 1; }

    __host__ __device__
    int totalNumGaussPoints() const { return numElements() * numGaussPoints(); }

    __device__
    double domainBegin() const { return m_knots[m_order]; }
    
    __device__
    double domainEnd() const { return m_knots[m_knots.size() - m_order - 1]; }

    __device__
    bool inDomain(double u) const
    { return u >= domainBegin() && u <= domainEnd(); }

    __host__ __device__
    int numKnots() const { return m_knots.size(); }

     __device__
    int upperBound(double value) const
    {
        int low = m_order - 1;
        int high = m_knots.size();
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (m_knots[mid] <= value)
                low = mid + 1;
            else
                high = mid;
        }
        return low;
    }

    __host__ __device__
    int numControlPoints() const { return m_knots.size() - m_order - 1; }
    
};

